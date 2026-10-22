"""Novita materialization of the per-task Terminal-Bench-2 sandbox recipe.

The recipe itself — the shell layers that turn a task's official image into a
combined task+env-server image — lives in ``tb2_sandbox_recipe`` (sibling
module) and is provider-agnostic. This module is everything Novita-specific
about turning that recipe into a running cloud sandbox on
`Novita Agent Sandbox <https://docs.novita.ai>`_.

Novita's shape is E2B's: a sandbox is created from a **named template**, so
the two halves are separate and the first can be run ahead of training:

  ``ensure_task_template(...)``  build the per-task template once, under a
      deterministic name derived from the task id AND a digest of the recipe
      itself — so editing the recipe re-bakes automatically instead of
      silently serving stale templates. The template's start command is
      pinned to a keepalive (``sleep infinity``): the SDK otherwise inherits
      the task image's own ENTRYPOINT/CMD as the start command, and what runs
      in this sandbox must be the recipe's decision, not the task image's.
  ``create_task_sandbox(...)``  per-episode: start a sandbox from the
      template, exec ``server_cmd()`` as root, wait for /health on the
      sandbox's public host, return ``(sandbox, base_url)``. The sandbox
      carries ownership metadata and a TTL armed as a dead-man's switch: a
      keepalive thread re-arms the timeout while the creating process lives,
      so a hard-killed caller's orphans are reclaimed instead of running
      forever.
  bake CLI (``python tb2_sandbox_novita.py ...``)  pre-build templates for a
      task list so the first training episode doesn't pay the image build.

The env server's URL is ``https://8000-<sandbox id>.<domain>``: Novita's
gateway terminates TLS and proxies HTTP and WebSocket alike, which is what
OpenEnv's client needs (health over HTTP, the episode over ``/ws``). The URL
carries no credential — the client has nowhere to put one — so the sandbox is
created with the default public service access and relies on the random
sandbox id plus its per-episode lifetime, exactly as the Modal backend does
with its tunnel URL. ``secure`` (the SDK's own control-plane token) is left at
the SDK default and is unrelated to that port.

All novita imports are lazy (in-function), mirroring the sibling provider
modules: offline unit tests and non-sandbox launches must not require the SDK.
"""

import argparse
import hashlib
import os
import re
import shlex
import sys
from pathlib import Path

import tb2_sandbox_recipe as recipe
from tb2_sandbox_recipe import (
    resolve_docker_image,
    run_with_deadline,
    sandbox_labels,
    server_cmd,
    server_layer_commands,
    task_env_resources,
    wait_server_ready,
)
from miles.rollout.agentic.credentials import resolve_provider_api_key
from miles.rollout.agentic.rollout_limits import rollout_lock


# The user every build command and the env server run as. The TB2 task images
# are built for a root agent (their solutions and tests apt-install freely), so
# anything less would change the task environment, not just the build.
#
# Novita's template parser defaults subsequent instructions to a non-root
# ``user`` (OpenEnv's own Novita provider documents the same), which fails
# every layer of the recipe (apt-get exits 100, /opt is not writable); and the
# env server executes the agent's commands, so the user it runs as IS the task
# environment's user.
_BUILD_USER = "root"

# The template's start command. Pinned so the sandbox comes up idle with port
# 8000 free; the env server is exec'd per episode (see create_task_sandbox) so
# runtime knobs stay runtime. The ready check is the SDK's fixed timer: there
# is nothing to probe for a sleep.
_KEEPALIVE_CMD = "sleep infinity"
_KEEPALIVE_READY_MS = 5_000

_ENV_SERVER_PORT = 8000


def template_name(task_dir: Path) -> str:
    """Deterministic template name: ``tb2-<task-id>-<recipe digest>``.

    The digest covers the base image, the build user, every build command (which
    embed the tbench2_env source, deterministically tarred — see
    ``_dir_tar_b64``), the pinned start command, and the build resources (Novita
    sizes sandboxes at template-build time), so the name changes exactly when the
    baked artifact would: recipe edits, env-package changes, or a task.toml
    resource bump re-bake; identical inputs reuse the existing template.
    """
    task_dir = Path(task_dir)
    base = resolve_docker_image(task_dir, None)
    resources = task_build_resources(task_dir)
    inputs = [
        base,
        _BUILD_USER,
        *server_layer_commands(task_dir),
        _KEEPALIVE_CMD,
        repr(sorted(resources.items())),
    ]
    digest = hashlib.sha256("\n".join(inputs).encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9-]", "-", task_dir.name.lower())
    return f"tb2-{slug}-{digest}"


def task_build_resources(task_dir: Path) -> dict[str, int]:
    """Template build resources from ``task.toml [environment]``.

    Novita sizes sandboxes at template-build time (a create inherits the
    template's spec), so the task's requirements go here, not on create.
    """
    cpus, memory_mb, _storage_mb = task_env_resources(task_dir)
    return {"cpu_count": cpus, "memory_mb": memory_mb}


def _connection_opts() -> dict:
    """Per-call connection kwargs. Only the key is passed explicitly (for the
    file indirection); the region and endpoint overrides (NOVITA_DOMAIN,
    NOVITA_API_URL, NOVITA_SANDBOX_URL) are read from the environment by the
    SDK itself."""
    return {"api_key": resolve_api_key()}


def ensure_task_template(
    task_dir: Path,
    *,
    force: bool = False,
    build_timeout_s: float = 1800.0,
    on_logs=None,
) -> str:
    """Build the per-task template if its name doesn't exist yet; return the name.

    *build_timeout_s* bounds the build's wall clock. ``Template.build`` blocks
    across many requests (trigger + log polling), so the deadline is enforced
    by running it on a scoped thread; on timeout the provider-side build keeps
    cooking (and may finish), but this caller stops holding the name lock and
    the create-semaphore slot.
    """
    # In-function import, deliberately: the novita SDK is an optional dependency
    # of this recipe (the launcher preflights it), so importing the module
    # must not require it — the offline tests import this file with a fake
    # `novita_sandbox` in sys.modules, and the sibling backends defer their SDKs
    # the same way.
    from novita_sandbox import Template, wait_for_timeout

    task_dir = Path(task_dir)
    name = template_name(task_dir)
    # one build per name per rollout: a rollout fans out many episodes of the same task at once,
    # and every concurrent miss would otherwise start its own multi-minute build
    with rollout_lock(f"openenv-novita-build-{name}"):
        if not force and Template.exists(name, **_connection_opts()):
            return name
        base = resolve_docker_image(task_dir, None)
        # set_user before the first command: the parser's default user is not
        # root, which fails every layer of the recipe (see _BUILD_USER).
        builder = Template().from_image(base).set_user(_BUILD_USER)
        for command in server_layer_commands(task_dir):
            builder = builder.run_cmd(command)
        # The start command must be set last: set_start_cmd returns the final,
        # buildable form of the template, which takes no further build steps.
        template = builder.set_start_cmd(_KEEPALIVE_CMD, wait_for_timeout(_KEEPALIVE_READY_MS))

        def _build() -> None:
            Template.build(
                template,
                name,
                **task_build_resources(task_dir),
                skip_cache=force,
                on_build_logs=on_logs,
                **_connection_opts(),
            )

        run_with_deadline(_build, build_timeout_s)
    return name


def base_url(sandbox) -> str:
    """The env server's externally reachable URL on port 8000.

    ``get_host`` yields the per-port hostname Novita's gateway routes
    (``{port}-{sandboxID}.{domain}``), behind TLS, so the scheme is always
    https — and the gateway proxies WebSocket as well as HTTP, which OpenEnv's
    client needs (health over HTTP, episode I/O over ``/ws``).

    Under NOVITA_DEBUG the SDK returns ``localhost:<port>`` instead: a URL that
    looks valid and silently targets whatever listens on the caller's machine.
    Refuse it rather than connect to the wrong thing.
    """
    host = str(sandbox.get_host(_ENV_SERVER_PORT))
    hostname = host.rsplit(":", 1)[0]
    if "://" in host or hostname in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError(
            f"Novita returned {host!r} as the sandbox host: a loopback host means the SDK is in "
            "NOVITA_DEBUG mode and would bypass the sandbox; unset NOVITA_DEBUG"
        )
    return f"https://{host}"


# TTL / keepalive: the sandbox is created with a bounded lifetime and the
# keepalive thread re-arms it ahead of expiry for as long as THIS process
# lives — the Novita timeout is kill-on-expiry, which is exactly a dead-man's
# switch for orphans. 6 beats per window; up to 3 consecutive failed beats
# (API blips) tolerated before the thread concludes the sandbox is gone.
#
# Novita caps one set_timeout at 1 hour unless the sandbox was created with the
# `long_running` metadata marker; the TTL here is a heartbeat window rather
# than an episode ceiling (any live episode keeps re-arming it), so the default
# stays well under that cap and a longer episode needs no marker.
_SANDBOX_TTL_S = int(os.getenv("OPENENV_NOVITA_SANDBOX_TTL_S", "1800"))
_READY_TIMEOUT_S = float(os.getenv("OPENENV_NOVITA_READY_TIMEOUT_S", "300"))
_KEEPALIVE_INTERVAL_S = _SANDBOX_TTL_S / 6.0
_KEEPALIVE_MAX_CONSECUTIVE_FAILURES = 3


def _start_keepalive(sandbox, task_id: str) -> None:
    """Re-arm the sandbox TTL for as long as THIS process lives.

    A live episode of any length keeps its sandbox; a hard-killed caller
    (SIGKILL, OOM, node loss) stops beating and the provider kills the
    sandbox within _SANDBOX_TTL_S of the last beat. The thread exits once
    re-arms fail persistently (the normal case: the episode ended and the
    caller killed the sandbox)."""
    opts = _connection_opts()
    recipe.start_keepalive(
        lambda: sandbox.set_timeout(_SANDBOX_TTL_S, **opts),
        f"tb2-sandbox-keepalive-{task_id}",
        interval_s=_KEEPALIVE_INTERVAL_S,
        max_consecutive_failures=_KEEPALIVE_MAX_CONSECUTIVE_FAILURES,
    )


def create_task_sandbox(
    task_dir: Path,
    *,
    command_timeout_s: int = recipe.COMMAND_TIMEOUT_S,
    ready_timeout_s: float = _READY_TIMEOUT_S,
):
    """Create ONE per-episode sandbox for *task_dir* from its template.

    Returns ``(sandbox, base_url)``. Caller must ``kill_sandbox(sandbox)`` when
    the episode ends. First create for a task pays the template build (via
    ``ensure_task_template``); repeats start from the built template.

    The template's start command is a keepalive, not the server: the server is
    exec'd here so runtime knobs (TB2_COMMAND_TIMEOUT_S) stay runtime — baking
    them would silently pin them until the next re-bake.
    """
    from novita_sandbox import Sandbox

    task_dir = Path(task_dir)
    opts = _connection_opts()
    name = ensure_task_template(task_dir)
    # The TTL is deliberately not a parameter: the keepalive thread re-arms
    # _SANDBOX_TTL_S, so a different value passed here would be silently
    # overwritten at the first beat.
    sandbox = Sandbox.create(
        template=name,
        timeout=_SANDBOX_TTL_S,
        metadata=sandbox_labels(task_dir),
        **opts,
    )
    try:
        cmd = server_cmd(command_timeout_s, default_task_id=task_dir.name)
        # user: the env server executes the agent's commands, so the user it
        # runs as IS the task environment's user (see _BUILD_USER). timeout=0:
        # the SDK's per-command deadline (60 s by default) must not apply to a
        # server that runs for the whole episode.
        sandbox.commands.run(
            f"bash -c {shlex.quote(cmd)} > /tmp/openenv-server.log 2>&1",
            background=True,
            user=_BUILD_USER,
            timeout=0,
        )
        url = base_url(sandbox)
        wait_server_ready(url, timeout_s=ready_timeout_s)
        _start_keepalive(sandbox, task_dir.name)
        return sandbox, url
    except Exception:
        try:
            sandbox.kill(**opts)
        except Exception:
            pass  # cleanup must not mask the real failure; the TTL reclaims it
        raise


def kill_sandbox(sandbox) -> None:
    """Kill a sandbox created here.

    The key is supplied again rather than trusted to the handle — which is this
    module's business, not its callers' — on the same reasoning as the sibling
    E2B module.
    """
    sandbox.kill(**_connection_opts())


_DEFAULT_API_KEY_FILE = "~/.config/novita/api_key"


def resolve_api_key() -> str:
    """The Novita API key: NOVITA_API_KEY, else the key file (see
    credentials.resolve_provider_api_key for the file-indirection rationale)."""
    return resolve_provider_api_key("NOVITA_API_KEY", "NOVITA_API_KEY_FILE", _DEFAULT_API_KEY_FILE)


def bake(tasks_dir: Path, task_id: str, force: bool) -> None:
    """Pre-build the template for one task (optional warm cache)."""
    task_dir = tasks_dir / task_id
    name = template_name(task_dir)
    resources = task_build_resources(task_dir)
    print(f"[bake] {name}  cpu={resources['cpu_count']} mem={resources['memory_mb']}MB")
    ensure_task_template(
        task_dir,
        force=force,
        on_logs=lambda entry: print(f"  | {entry}", flush=True),
    )
    print(f"[done] {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    recipe.add_task_selection_args(ap)
    ap.add_argument("--force", action="store_true", help="rebuild even when the template exists")
    args = ap.parse_args()

    tasks_dir, task_ids = recipe.selected_task_ids(args)

    for task_id in task_ids:
        bake(tasks_dir, task_id, args.force)


if __name__ == "__main__":
    sys.exit(main())
