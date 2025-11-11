from faker import Faker

from scribae.project import ProjectConfig
from scribae.prompts import SYSTEM_PROMPT, build_prompt_bundle, build_user_prompt


def test_build_user_prompt_includes_project_details(fake: Faker) -> None:
    site_name = fake.company()
    domain = fake.url()
    audience = fake.sentence(nb_words=3)
    tone = fake.word()
    keywords = [fake.word(), fake.word()]

    project: ProjectConfig = {
        "site_name": site_name,
        "domain": domain,
        "audience": audience,
        "tone": tone,
        "keywords": keywords,
        "language": "de",
    }

    note_title = fake.sentence(nb_words=3)
    note_content = fake.paragraph()

    prompt = build_user_prompt(project=project, note_title=note_title, note_content=note_content)

    assert f"Site: {site_name} ({domain})" in prompt
    assert f"FocusKeywords: {', '.join(keywords)}" in prompt
    assert f"[NOTE TITLE]\n{note_title}" in prompt
    assert f"[NOTE CONTENT]\n{note_content}" in prompt


def test_build_prompt_bundle_uses_system_prompt(fake: Faker) -> None:
    project: ProjectConfig = {
        "site_name": fake.company(),
        "domain": fake.url(),
        "audience": fake.sentence(nb_words=3),
        "tone": fake.word(),
        "keywords": [],
        "language": "en",
    }

    bundle = build_prompt_bundle(project=project, note_title=fake.sentence(nb_words=3), note_content=fake.paragraph())

    assert bundle.system_prompt == SYSTEM_PROMPT
    assert "[TASK]" in bundle.user_prompt
    assert "FocusKeywords: none" in bundle.user_prompt
