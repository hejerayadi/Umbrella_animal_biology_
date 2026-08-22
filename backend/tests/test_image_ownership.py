from backend.image_store import ImageStore


PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"


def test_images_are_only_returned_to_their_owner() -> None:
    store = ImageStore()
    stored = store.add(PNG, "sample.png", owner_id="user-a")
    assert store.get_for_owner(stored.image_id, "user-a") is stored
    assert store.get_for_owner(stored.image_id, "user-b") is None

