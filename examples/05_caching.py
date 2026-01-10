"""Prompt caching example with FlowPrompt.

This example demonstrates how to use FlowPrompt's caching system
to reduce API costs and improve response times.
"""

from flowprompt import Prompt, configure_cache, get_cache


class TranslatePrompt(Prompt):
    """Translate text between languages."""

    system = "You are an expert translator. Translate accurately and naturally."
    user = "Translate '{text}' from {source_lang} to {target_lang}."


def demonstrate_caching() -> None:
    """Show how caching works."""
    # Configure cache with 1-hour TTL
    configure_cache(enabled=True, default_ttl=3600)

    cache = get_cache()
    print("Cache configured:")
    print(f"  Enabled: {cache.stats['enabled']}")
    print(f"  Initial size: {cache.stats['size']}")

    # Create identical prompts
    prompt1 = TranslatePrompt(
        text="Hello, world!",
        source_lang="English",
        target_lang="Spanish",
    )
    prompt2 = TranslatePrompt(
        text="Hello, world!",
        source_lang="English",
        target_lang="Spanish",
    )

    # Prompts have the same content hash
    print(f"\nPrompt 1 hash: {prompt1.content_hash}")
    print(f"Prompt 2 hash: {prompt2.content_hash}")
    print(f"Hashes match: {prompt1.content_hash == prompt2.content_hash}")

    # First call hits the API (cache miss)
    # result1 = prompt1.run(model="gpt-4o")  # API call
    # print(f"\nFirst call result: {result1}")

    # Second call uses cache (cache hit!)
    # result2 = prompt2.run(model="gpt-4o")  # From cache - no API call
    # print(f"Second call result: {result2}")

    # View cache stats
    print("\nCache statistics:")
    stats = cache.stats
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Size: {stats['size']} entries")


def file_based_cache() -> None:
    """Demonstrate persistent file-based caching."""
    from flowprompt import FileCache

    print("\nFile-based cache (persistent across sessions):")
    print("-" * 50)

    # Configure with file backend
    configure_cache(
        enabled=True,
        backend=FileCache(".flowprompt_cache"),
        default_ttl=86400,  # 24 hours
    )

    print("  Cache directory: .flowprompt_cache/")
    print("  TTL: 24 hours")
    print("  Cache persists between program runs")


def main() -> None:
    """Run the caching examples."""
    print("FlowPrompt Caching Example")
    print("=" * 50)

    demonstrate_caching()
    file_based_cache()

    print("\nCaching benefits:")
    print("  - 50-90% cost reduction on repeated prompts")
    print("  - Instant responses for cached queries")
    print("  - Persistent cache survives restarts")


if __name__ == "__main__":
    main()
