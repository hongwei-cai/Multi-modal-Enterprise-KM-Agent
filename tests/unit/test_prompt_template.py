from src.rag.prompt_template import PromptTemplate, get_prompt_template


def test_prompt_template_init():
    """Test template initialization."""
    template = PromptTemplate()
    assert "You are a helpful assistant" in template.system_role


def test_format_prompt():
    """Test prompt formatting."""
    template = PromptTemplate()
    context_docs = [
        {"document": "This is context 1."},
        {"document": "This is context 2."},
    ]
    prompt = template.format_prompt("What is AI?", context_docs)
    assert "Context:" in prompt
    assert "Question: What is AI?" in prompt
    assert "This is context 1." in prompt


def test_format_prompt_truncation():
    """Test context truncation."""
    template = PromptTemplate()
    long_context = [{"document": "A" * 3000}]
    prompt = template.format_prompt("Test?", long_context, max_context_length=500)
    assert "..." in prompt
    assert len(prompt) < 800  # Approximate


def test_get_prompt_template():
    """Test convenience function."""
    template = get_prompt_template(system_role="Custom role")
    assert "Custom role" in template.system_role
