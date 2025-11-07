from scribae.main import main


def test_main_prints_hello(capsys):
    """Test that main() prints the expected message."""
    main()

    captured = capsys.readouterr()
    assert captured.out == "Hello from scribae!\n"


def test_main_runs_without_error():
    """Test that main() executes without raising an exception."""
    main()
