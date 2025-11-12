import nox

nox.options.sessions = ["tests"]


@nox.session()
def tests(session: nox.Session) -> None:
    """Run unit tests in an isolated virtual environment."""
    session.install("pip", "setuptools>=69", "wheel")
    session.install(".[dev]")
    session.run("pytest")

