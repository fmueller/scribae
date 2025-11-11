from collections.abc import Generator

import pytest
from faker import Faker


@pytest.fixture()
def fake() -> Generator[Faker]:
    faker = Faker()
    # Ensure deterministic data per test function
    faker.seed_instance(1337)
    yield faker
