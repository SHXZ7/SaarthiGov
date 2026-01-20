"use client";

import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [service, setService] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          top_k: 5,
          include_sources: true,
          service: service || null,
        }),
      });

      const data = await res.json();
      setAnswer(data.answer || "");
      setSources(data.sources || []);
    } catch (err) {
      console.error("Error fetching results:", err);
      setAnswer("Error connecting to the server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl text-black font-bold mb-4">
          Kerala Government Services Assistant (Demo)
        </h1>

        <p className="text-black mb-6">
          Ask a question about government services (English or Malayalam).
        </p>

        <div className="flex gap-2 mb-4">
          <select
            value={service}
            onChange={(e) => setService(e.target.value)}
            className="border border-gray-300 rounded px-3 py-2 bg-white text-gray-700"
          >
            <option value="">All Services</option>
            <option value="ration_card">Ration Card</option>
            <option value="birth_certificate">Birth Certificate</option>
            <option value="unemployment_allowance">Unemployment Allowance</option>
          </select>
        </div>

        <div className="flex gap-2 text-black mb-6">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="e.g. How to apply for birth certificate online?"
            className="flex-1 border border-gray-300 rounded px-3 py-2"
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {loading && <p className="text-gray-500">Thinking…</p>}

        {answer && (
          <div className="mb-6">
            <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h2 className="font-semibold text-lg text-gray-800">Answer</h2>
                {service && (
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                    {service === "ration_card" ? "Ration Card" : 
                     service === "birth_certificate" ? "Birth Certificate" : 
                     "Unemployment Allowance"}
                  </span>
                )}
              </div>
              <div className="prose prose-sm text-gray-700 whitespace-pre-wrap">
                {answer}
              </div>
            </div>
          </div>
        )}

        {sources.length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-medium text-gray-500 mb-3">Sources</h3>
            <div className="space-y-3">
              {sources.map((item, idx) => (
                <details
                  key={idx}
                  className="bg-gray-100 border border-gray-200 rounded p-3"
                >
                  <summary className="cursor-pointer text-sm text-blue-700 font-medium">
                    {item.section} ({item.service}) — score: {item.score.toFixed(3)}
                  </summary>
                  <pre className="whitespace-pre-wrap text-xs text-gray-600 mt-2">
                    {item.text}
                  </pre>
                </details>
              ))}
            </div>
          </div>
        )}

        {!answer && !loading && (
          <p className="text-black mt-6">
            No results yet. Ask a question to see retrieved government content.
          </p>
        )}
      </div>
    </main>
  );
}
