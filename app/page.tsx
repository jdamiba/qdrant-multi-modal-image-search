import dynamic from "next/dynamic";
import Image from "next/image";

// Import components with dynamic loading to prevent hydration errors and enable code splitting
const ImageSearch = dynamic(() => import("../app/components/ImageSearch"), {
  ssr: false,
  loading: () => (
    <div className="animate-pulse h-96 w-full bg-gray-100 rounded-lg flex items-center justify-center">
      <div className="text-gray-500">Loading search component...</div>
    </div>
  ),
});

// Add page metadata for better performance
export const metadata = {
  title: "House Image Search - Find Similar Houses",
  description:
    "Search for houses by describing features and find visually similar properties.",
};

export default function Home() {
  return (
    <div className="w-full">
      {/* Hero section */}
      <section className="bg-background border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 md:py-12">
          <div className="text-center">
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground mb-2">
              <a
                href="https://qdrant.tech"
                target="_blank"
                rel="noopener noreferrer"
                className="text-amaranth hover:text-amaranth/80"
              >
                Qdrant
              </a>{" "}
              Multi-Modal Home Search
            </h1>
            <h2 className="text-lg md:text-xl font-medium text-gray-500 mb-3">
              Visual Property Search Engine
            </h2>
            <div className="max-w-3xl mx-auto mb-4">
              <p className="text-sm text-gray-600 italic">
                Find your dream home by describing what you're looking for —
                from Victorian architecture to modern interiors, our AI-powered
                search understands your preferences and finds visually similar
                properties.
              </p>
            </div>
            <p className="max-w-4xl mx-auto text-md text-gray-600">
              Search our database of house images using natural language powered
              by{" "}
              <a
                href="https://qdrant.tech"
                target="_blank"
                rel="noopener noreferrer"
                className="text-amaranth font-medium hover:underline"
              >
                Qdrant
              </a>{" "}
              vector search and{" "}
              <a
                href="https://openai.com/research/clip"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#2F6FF0] hover:underline"
              >
                CLIP
              </a>{" "}
              embeddings.
            </p>
          </div>
        </div>
      </section>

      {/* Main app section */}
      <section className="py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="card px-6 pt-6 pb-16 bg-background border border-gray-200 rounded-lg shadow-sm flex flex-col">
            <ImageSearch />
          </div>
        </div>
      </section>

      {/* Features section */}
      <section className="py-12 bg-background border-y border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">
              How It <span className="text-amaranth">Works</span>
            </h2>
            <p className="max-w-2xl mx-auto text-gray-600">
              Our house search engine uses advanced AI technologies to
              understand both images and text descriptions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="relative">
              <div className="card p-6 h-full flex flex-col bg-background border border-gray-200 rounded-lg shadow-sm">
                <span className="absolute -top-4 -left-4 w-12 h-12 bg-amaranth text-background rounded-full flex items-center justify-center text-xl font-semibold shadow-md">
                  1
                </span>
                <h3 className="text-xl font-semibold text-foreground mb-3 mt-3">
                  Image Understanding
                </h3>
                <p className="text-gray-600 flex-grow">
                  Our system analyzes thousands of house images, understanding
                  architectural styles, interior features, and design elements.
                </p>
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <span className="text-sm text-amaranth">
                    Powered by CLIP Vision Model
                  </span>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="card p-6 h-full flex flex-col bg-background border border-gray-200 rounded-lg shadow-sm">
                <span className="absolute -top-4 -left-4 w-12 h-12 bg-amaranth text-background rounded-full flex items-center justify-center text-xl font-semibold shadow-md">
                  2
                </span>
                <h3 className="text-xl font-semibold text-foreground mb-3 mt-3">
                  Vector Embeddings
                </h3>
                <p className="text-gray-600 flex-grow">
                  Images are converted into vector embeddings that capture
                  visual features, while your text queries are converted to the
                  same vector space.
                </p>
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <span className="text-sm text-amaranth">
                    Using OpenAI's CLIP technology
                  </span>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="card p-6 h-full flex flex-col bg-background border border-gray-200 rounded-lg shadow-sm">
                <span className="absolute -top-4 -left-4 w-12 h-12 bg-amaranth text-background rounded-full flex items-center justify-center text-xl font-semibold shadow-md">
                  3
                </span>
                <h3 className="text-xl font-semibold text-foreground mb-3 mt-3">
                  Semantic Search
                </h3>
                <p className="text-gray-600 flex-grow">
                  Find houses by describing what you want in natural language.
                  Our system matches your description with visually similar
                  properties in our database.
                </p>
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <span className="text-sm text-amaranth">
                    Built with Qdrant vector database
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Search examples section */}
      <section className="py-12 md:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">
              Example <span className="text-amaranth">Searches</span>
            </h2>
            <p className="max-w-2xl mx-auto text-gray-600">
              Try these example searches to see how our system works:
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="card p-8 border border-gray-200 rounded-lg shadow-sm bg-background hover:border-l-4 hover:border-l-amaranth transition-all">
              <div className="flex items-center mb-4">
                <svg
                  className="h-8 w-8 text-amaranth mr-3"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
                  />
                </svg>
                <h3 className="text-xl font-semibold text-foreground">
                  Architectural Styles
                </h3>
              </div>
              <p className="text-gray-600 mb-4">
                Try searching for specific architectural styles like "Victorian
                house", "Modern farmhouse", "Mid-century modern home", or
                "Mediterranean villa".
              </p>
              <span className="badge badge-amaranth inline-flex px-2 py-1 text-xs font-medium rounded-md">
                Style
              </span>
            </div>

            <div className="card p-8 border border-gray-200 rounded-lg shadow-sm bg-background hover:border-l-4 hover:border-l-amaranth transition-all">
              <div className="flex items-center mb-4">
                <svg
                  className="h-8 w-8 text-amaranth mr-3"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"
                  />
                </svg>
                <h3 className="text-xl font-semibold text-foreground">
                  Features & Colors
                </h3>
              </div>
              <p className="text-gray-600 mb-4">
                Search for specific features like "house with red brick", "home
                with large windows", "house with swimming pool", or "kitchen
                with marble countertops".
              </p>
              <span className="badge badge-amaranth inline-flex px-2 py-1 text-xs font-medium rounded-md">
                Features
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
