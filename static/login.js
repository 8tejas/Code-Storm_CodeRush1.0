// Toggle password visibility
function togglePassword() {
  let pwd = document.getElementById("password");
  pwd.type = (pwd.type === "password") ? "text" : "password";
}

// Load saved data when page loads
window.addEventListener("DOMContentLoaded", () => {
  const savedEmail = localStorage.getItem("email");
  const savedPassword = localStorage.getItem("password");

  if (savedEmail) {
    document.querySelector("input[type='email']").value = savedEmail;
  }
  if (savedPassword) {
    document.querySelector("input[type='password']").value = savedPassword;
  }
});

// Save data on form submit
document.querySelector("form").addEventListener("submit", (e) => {
  e.preventDefault(); // prevent actual submit (for demo)

  const email = document.querySelector("input[type='email']").value;
  const password = document.querySelector("input[type='password']").value;

  // Save in localStorage
  localStorage.setItem("email", email);
  localStorage.setItem("password", password);

  alert("Login details saved! (Next time, they’ll be remembered)");
});
