function toggleTheme() {
    const html = document.documentElement;
    const themeToggleIcon = document.querySelector(".theme-toggle i");
  
    if (html.getAttribute("data-theme") === "light") {
      html.setAttribute("data-theme", "dark");
      themeToggleIcon.classList.remove("fa-moon");
      themeToggleIcon.classList.add("fa-sun");
      localStorage.setItem("theme", "dark");
    } else {
      html.setAttribute("data-theme", "light");
      themeToggleIcon.classList.remove("fa-sun");
      themeToggleIcon.classList.add("fa-moon");
      localStorage.setItem("theme", "light");
    }
  }
  
  document.addEventListener("DOMContentLoaded", function () {
    const html = document.documentElement;
    const themeToggleIcon = document.querySelector(".theme-toggle i");
    const savedTheme = localStorage.getItem("theme") || "light";
  
    html.setAttribute("data-theme", savedTheme);
    if (savedTheme === "dark") {
      themeToggleIcon.classList.remove("fa-moon");
      themeToggleIcon.classList.add("fa-sun");
    } else {
      themeToggleIcon.classList.remove("fa-sun");
      themeToggleIcon.classList.add("fa-moon");
    }
  });
  
  function scrollToBottom() {
    const container = document.getElementById("chatContainer");
    container.scrollTop = container.scrollHeight;
  }
  
  window.onload = scrollToBottom;
  
  const observer = new MutationObserver(scrollToBottom);
  observer.observe(document.getElementById("chatContainer"), {
    childList: true,
    subtree: true,
  });
  
  // Add message input animations
  const messageInput = document.querySelector('input[type="text"]');
  const sendButton = document.querySelector(".send-button");
  
  messageInput.addEventListener("input", function () {
    if (this.value.trim()) {
      sendButton.style.transform = "scale(1.1)";
    } else {
      sendButton.style.transform = "scale(1)";
    }
  });
  
  