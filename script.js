const copyButton = document.getElementById('copyBib');
copyButton?.addEventListener('click', async () => {
  const text = document.getElementById('bibtex').innerText;
  try {
    await navigator.clipboard.writeText(text);
    const old = copyButton.textContent;
    copyButton.textContent = 'Copied';
    setTimeout(() => copyButton.textContent = old, 1200);
  } catch (_) {
    copyButton.textContent = 'Select & copy';
  }
});
