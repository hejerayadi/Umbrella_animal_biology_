from app import cli


def test_cli_uses_selector_event_loop_policy_on_windows(monkeypatch) -> None:
    policy = object()
    configured_policies: list[object] = []

    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        cli.asyncio,
        "WindowsSelectorEventLoopPolicy",
        lambda: policy,
        raising=False,
    )
    monkeypatch.setattr(cli.asyncio, "set_event_loop_policy", configured_policies.append)

    cli.configure_event_loop_policy()

    assert configured_policies == [policy]
