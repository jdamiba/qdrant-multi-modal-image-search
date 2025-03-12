"use client";

import { useState } from "react";
import Image from "next/image";

interface SearchResult {
  image_path: string;
  filename: string;
  score: number;
}

export default function ImageSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query.trim()) {
      setError("Please enter a search query");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `/api/py/search?query=${encodeURIComponent(query)}&limit=20`
      );

      if (!response.ok) {
        throw new Error("Search failed. Please try again.");
      }

      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      console.error("Search error:", err);
      setError("An error occurred while searching. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Example search suggestions
  const searchSuggestions = [
    "Victorian house with bay windows",
    "Modern minimalist kitchen",
    "Farmhouse with red barn",
    "Coastal beach house",
    "House with swimming pool",
    "Rustic cabin in the woods",
  ];

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    // Submit the form programmatically
    document
      .getElementById("search-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
  };

  return (
    <div className="w-full">
      <h2 className="text-2xl font-bold mb-6">Find Similar Houses</h2>

      <form id="search-form" onSubmit={handleSearch} className="mb-8">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-grow">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe the house you're looking for..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amaranth focus:border-transparent"
            />
          </div>
          <button
            type="submit"
            className="bg-amaranth hover:bg-amaranth/90 text-white font-medium py-3 px-6 rounded-lg transition-colors"
            disabled={loading}
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {error && <div className="mt-2 text-red-500 text-sm">{error}</div>}
      </form>

      {/* Search suggestions */}
      <div className="mb-8">
        <h3 className="text-sm font-medium text-gray-500 mb-2">
          Try searching for:
        </h3>
        <div className="flex flex-wrap gap-2">
          {searchSuggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => handleSuggestionClick(suggestion)}
              className="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 px-3 py-1 rounded-full transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold mb-4">Search Results</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {results.map((result, index) => (
              <div
                key={index}
                className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow"
              >
                <div className="relative h-48 w-full">
                  <Image
                    src={`/images/${result.image_path}`}
                    alt={result.filename}
                    fill
                    style={{ objectFit: "cover" }}
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                  />
                </div>
                <div className="p-3">
                  <p
                    className="text-sm text-gray-500 truncate"
                    title={result.filename}
                  >
                    {result.filename}
                  </p>
                  <p className="text-xs text-gray-400">
                    Score: {(result.score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="flex justify-center my-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-amaranth"></div>
        </div>
      )}

      {!loading && results.length === 0 && query && (
        <div className="text-center py-8">
          <p className="text-gray-500">
            No results found. Try a different search query.
          </p>
        </div>
      )}
    </div>
  );
}
