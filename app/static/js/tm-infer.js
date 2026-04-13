(() => {
  const configEl = document.getElementById("tm-config");
  const modelUrl = configEl?.dataset?.modelUrl;
  const metadataUrl = configEl?.dataset?.metadataUrl;
  const storePredictions = configEl?.dataset?.storePredictions === "true";
  const topPredictions = Number(configEl?.dataset?.topPredictions || 5);

  const form = document.querySelector(".upload-form");
  const runButton = document.getElementById("run-analysis");
  const fileInput = document.getElementById("image");
  const errorEl = document.getElementById("client-error");
  const statusEl = document.getElementById("model-status");
  const resultsEl = document.getElementById("results");
  const predictionsList = document.getElementById("predictions-list");
  const predictionsNote = document.getElementById("predictions-note");
  const previewImg = document.getElementById("image-preview");
  const filenameEl = document.getElementById("image-filename");
  const chipSample = document.getElementById("chip-sample-type");
  const chipLocation = document.getElementById("chip-location");
  const notesEl = document.getElementById("notes-text");
  const diagLabel = document.getElementById("diagnosis-label");
  const diagMeta = document.getElementById("diagnosis-meta");
  const diagPlaceholder = document.getElementById("diagnosis-placeholder");
  const recTitle = document.getElementById("recommendations-title");
  const recList = document.getElementById("recommendations-list");

  if (!form || !runButton || !fileInput) {
    return;
  }

  let modelPromise = null;
  let previewUrl = null;

  const setStatus = (message) => {
    if (statusEl) {
      statusEl.textContent = message || "";
    }
  };

  const showError = (message) => {
    if (!errorEl) {
      return;
    }
    errorEl.textContent = message;
    errorEl.classList.remove("hidden");
  };

  const clearError = () => {
    if (!errorEl) {
      return;
    }
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
  };

  const resetDiagnosis = () => {
    if (diagPlaceholder) {
      diagPlaceholder.classList.remove("hidden");
    }
    if (diagLabel) {
      diagLabel.textContent = "";
      diagLabel.classList.add("hidden");
    }
    if (diagMeta) {
      diagMeta.textContent = "";
      diagMeta.classList.add("hidden");
    }
    if (recTitle) {
      recTitle.classList.add("hidden");
    }
    if (recList) {
      recList.innerHTML = "";
      recList.classList.add("hidden");
    }
  };

  const loadModel = async () => {
    if (!modelUrl || !metadataUrl) {
      throw new Error("Model URLs are not configured.");
    }
    if (!window.tmImage || !window.tmImage.load) {
      throw new Error("Teachable Machine library failed to load.");
    }
    if (!modelPromise) {
      setStatus("Loading model...");
      modelPromise = window.tmImage.load(modelUrl, metadataUrl);
    }
    const model = await modelPromise;
    setStatus("Model ready.");
    return model;
  };

  const updatePreview = (file) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    filenameEl.textContent = file.name || "upload";

    const sampleType = document.getElementById("sample_type")?.value || "leaf";
    chipSample.textContent = sampleType.charAt(0).toUpperCase() + sampleType.slice(1);

    const locationValue = document.getElementById("location")?.value?.trim();
    if (locationValue) {
      chipLocation.textContent = locationValue;
      chipLocation.classList.remove("hidden");
    } else {
      chipLocation.textContent = "";
      chipLocation.classList.add("hidden");
    }

    const notesValue = document.getElementById("notes")?.value?.trim();
    if (notesValue) {
      notesEl.textContent = notesValue;
      notesEl.classList.remove("hidden");
    } else {
      notesEl.textContent = "";
      notesEl.classList.add("hidden");
    }

    resultsEl.classList.remove("hidden");
  };

  const waitForImage = () =>
    new Promise((resolve, reject) => {
      if (previewImg.complete && previewImg.naturalWidth > 0) {
        resolve();
        return;
      }
      const onLoad = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error("Unable to read the uploaded image."));
      };
      const cleanup = () => {
        previewImg.removeEventListener("load", onLoad);
        previewImg.removeEventListener("error", onError);
      };
      previewImg.addEventListener("load", onLoad);
      previewImg.addEventListener("error", onError);
    });

  const renderPredictions = (predictions) => {
    predictionsList.innerHTML = "";
    const limited = predictions.slice(0, topPredictions);
    limited.forEach((prediction) => {
      const li = document.createElement("li");
      const line = document.createElement("div");
      line.className = "prediction-line";

      const label = document.createElement("strong");
      label.textContent = prediction.label;

      const confidence = document.createElement("span");
      confidence.textContent = `${(prediction.confidence * 100).toFixed(2)}%`;

      line.appendChild(label);
      line.appendChild(confidence);

      const progress = document.createElement("div");
      progress.className = "progress";
      const bar = document.createElement("div");
      bar.className = "progress-bar";
      bar.style.width = `${(prediction.confidence * 100).toFixed(2)}%`;
      progress.appendChild(bar);

      li.appendChild(line);
      li.appendChild(progress);
      predictionsList.appendChild(li);
    });

    if (predictionsNote) {
      if (storePredictions) {
        predictionsNote.classList.remove("hidden");
      } else {
        predictionsNote.classList.add("hidden");
      }
    }
  };

  const renderDiagnosis = (diagnosis) => {
    if (!diagnosis) {
      resetDiagnosis();
      return;
    }
    if (diagPlaceholder) {
      diagPlaceholder.classList.add("hidden");
    }
    if (diagLabel) {
      diagLabel.textContent = diagnosis.label;
      diagLabel.classList.remove("hidden");
    }
    if (diagMeta) {
      diagMeta.textContent = `Confidence ${(diagnosis.confidence * 100).toFixed(1)}%`;
      diagMeta.classList.remove("hidden");
    }
  };

  const renderRecommendations = (recommendations) => {
    if (!recList || !recTitle) {
      return;
    }
    recList.innerHTML = "";
    if (!recommendations || recommendations.length === 0) {
      recTitle.classList.add("hidden");
      recList.classList.add("hidden");
      return;
    }
    recTitle.classList.remove("hidden");
    recList.classList.remove("hidden");
    recommendations.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      recList.appendChild(li);
    });
  };

  const sendForRecommendations = async (payload) => {
    const response = await fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error("Recommendation service unavailable.");
    }
    return response.json();
  };

  const runAnalysis = async () => {
    clearError();
    resetDiagnosis();

    const file = fileInput.files?.[0];
    if (!file) {
      showError("Choose an image before running prediction.");
      return;
    }

    updatePreview(file);

    runButton.disabled = true;
    runButton.textContent = "Running...";

    try {
      const model = await loadModel();
      await waitForImage();
      const rawPredictions = await model.predict(previewImg);
      const predictions = rawPredictions
        .map((item) => ({
          label: item.className,
          confidence: item.probability,
        }))
        .sort((a, b) => b.confidence - a.confidence);

      renderPredictions(predictions);
      renderDiagnosis(predictions[0]);

      const sampleType = document.getElementById("sample_type")?.value || "leaf";
      const locationValue = document.getElementById("location")?.value?.trim();
      const notesValue = document.getElementById("notes")?.value?.trim();

      try {
        const response = await sendForRecommendations({
          predictions,
          sample_type: sampleType,
          filename: file.name,
          mime_type: file.type || "image/jpeg",
          location: locationValue,
          notes: notesValue,
        });
        renderDiagnosis(response.diagnosis || predictions[0]);
        renderRecommendations(response.recommendations || []);
      } catch (err) {
        renderRecommendations([]);
      }
    } catch (err) {
      showError(err.message || "Unable to run inference right now.");
    } finally {
      runButton.disabled = false;
      runButton.textContent = "Run Diagnosis";
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
  });
  runButton.addEventListener("click", runAnalysis);
})();
