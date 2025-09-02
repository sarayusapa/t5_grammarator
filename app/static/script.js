async function correctSingle() {
    const text = document.getElementById("singleInput").value;
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    document.getElementById("singleOutput").innerText = data.output;
  }

  async function correctBatch() {
    const input = document.getElementById("batchInput").value;
    const texts = input.split("\n").map(s => s.trim()).filter(Boolean);

    const res = await fetch("/predict_batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts })
    });
    const data = await res.json();

    document.getElementById("batchOutput").innerText = data.outputs.join("\n");
  }