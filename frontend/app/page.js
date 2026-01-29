"use client";

import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [service, setService] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
  if (!query.trim()) return;

  const userMsg = {
    role: "user",
    content: query,
  };

  // show user message immediately
  setMessages((prev) => [...prev, userMsg]);
  setLoading(true);

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
        service: service ? service : null,
        history: [...messages, userMsg], // 🔑 conversational memory
      }),
    });

    const data = await res.json();

    const botMsg = {
      role: "assistant",
      content: data.answer || "No answer generated.",
      sources: data.sources || [],
      next_steps: data.next_steps || [], // ✅ ADD THIS
    };

    setMessages((prev) => [...prev, botMsg]);
  } catch (err) {
    console.error(err);
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "Error connecting to server.",
      },
    ]);
  } finally {
    setLoading(false);
    setQuery("");
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
            <option value="">Auto-Detect</option>
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
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="e.g. How to apply for birth certificate online?"
            className="flex-1 border border-gray-300 rounded px-3 py-2"
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {loading && <p className="text-gray-500">Thinking…</p>}

<div className="space-y-4 mb-6">
  {messages.map((msg, idx) => (
    <div
      key={idx}
      className={`p-4 rounded-lg ${
        msg.role === "user"
          ? "bg-blue-600 text-white ml-auto max-w-[80%]"
          : "bg-white border text-gray-800 max-w-[80%]"
      }`}
    >
      <div className="whitespace-pre-wrap text-sm">
        {msg.content}
        {/* Next-step recommendations (only for assistant messages) */}
{msg.role === "assistant" && msg.next_steps?.length > 0 && (
  <div className="mt-3 text-sm">
    <p className="font-semibold text-gray-700">
      Recommended next steps
    </p>
    <ul className="list-disc ml-5 text-gray-600">
      {msg.next_steps.map((step, i) => (
        <li key={i}>
          {step.replace("_", " ")}
        </li>
      ))}
    </ul>
  </div>
)}

      </div>

      {/* Sources (only for assistant messages) */}
      {msg.role === "assistant" && msg.sources?.length > 0 && (
        <div className="mt-3">
          <p className="text-xs text-gray-500 mb-1">Sources</p>
          {msg.sources.map((s, i) => (
            <details key={i} className="text-xs text-gray-600 mb-1">
              <summary className="cursor-pointer">
                {s.section} — score {s.score.toFixed(3)}
              </summary>
              <pre className="whitespace-pre-wrap mt-1">
                {s.text}
              </pre>
            </details>
          ))}
        </div>
      )}
    </div>
  ))}
</div>


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
