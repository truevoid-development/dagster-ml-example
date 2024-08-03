import dagster


def persistent_run_id(context: dagster.OpExecutionContext | dagster.InitResourceContext) -> str:
    """Return a run identifier that does not change regardless of re-execution."""

    if hasattr(context, "run"):
        run = context.run
    elif hasattr(context, "dagster_run"):
        run = context.dagster_run
    else:
        raise NotImplementedError

    return run.root_run_id if run.root_run_id is not None else context.run_id
