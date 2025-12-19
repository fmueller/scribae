from faker import Faker

from scribae.brief import FaqItem, SeoBrief
from scribae.idea import Idea
from scribae.project import ProjectConfig
from scribae.prompts.brief import SYSTEM_PROMPT, build_prompt_bundle, build_user_prompt
from scribae.prompts.write import build_user_prompt as build_writer_prompt


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
        "allowed_tags": None,
    }

    note_title = fake.sentence(nb_words=3)
    note_content = fake.paragraph()

    prompt = build_user_prompt(
        project=project, note_title=note_title, note_content=note_content, language=project["language"]
    )

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
        "allowed_tags": None,
    }

    bundle = build_prompt_bundle(
        project=project,
        note_title=fake.sentence(nb_words=3),
        note_content=fake.paragraph(),
        language=project["language"],
    )

    assert bundle.system_prompt == SYSTEM_PROMPT
    assert "[TASK]" in bundle.user_prompt
    assert "FocusKeywords: none" in bundle.user_prompt
    assert "Output directive: write the entire brief in language code" in bundle.user_prompt


def test_writer_prompt_contains_language_directive(fake: Faker) -> None:
    project: ProjectConfig = {
        "site_name": fake.company(),
        "domain": fake.url(),
        "audience": fake.sentence(nb_words=3),
        "tone": fake.word(),
        "keywords": [],
        "language": "es-ES",
        "allowed_tags": None,
    }

    brief = SeoBrief(
        primary_keyword="alpha",
        secondary_keywords=["beta"],
        search_intent="informational",
        audience="Audience text",
        angle="Angle text",
        title="Title text",
        h1="Heading",
        outline=["One", "Two", "Three", "Four", "Five", "Six"],
        faq=[
            FaqItem(question="What is this?", answer="This is a sufficiently long answer for testing."),
            FaqItem(
                question="Why now?",
                answer="Because validation expects at least two entries with real text.",
            ),
        ],
        meta_description="A sufficiently long meta description for testing language prompts.",
    )
    prompt = build_writer_prompt(
        project=project,
        brief=brief,
        section_title="Intro",
        note_snippets="content",
        evidence_required=False,
        language="es-ES",
    )

    assert "write this section in language code 'es-ES'" in prompt


def test_build_user_prompt_includes_idea_block(fake: Faker) -> None:
    project: ProjectConfig = {
        "site_name": fake.company(),
        "domain": fake.url(),
        "audience": fake.sentence(nb_words=3),
        "tone": fake.word(),
        "keywords": [],
        "language": "en",
        "allowed_tags": None,
    }

    idea = Idea(
        id=fake.slug(),
        title=fake.sentence(nb_words=6),
        description=fake.paragraph(),
        why=fake.sentence(),
    )

    prompt = build_user_prompt(
        project=project,
        note_title=fake.sentence(nb_words=3),
        note_content=fake.paragraph(),
        language=project["language"],
        idea=idea,
    )

    assert "[IDEA]" in prompt
    assert idea.title in prompt
    assert idea.id in prompt
