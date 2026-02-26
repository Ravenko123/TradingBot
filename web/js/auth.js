const API_BASE_URL = 'http://localhost:5000/api';

function setStatus(el, message, isError = false) {
    if (!el) return;
    el.textContent = message;
    el.className = isError ? 'auth-status error' : 'auth-status success';
}

async function signup(name, email, password) {
    const res = await fetch(`${API_BASE_URL}/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password })
    });
    return res.json();
}

async function login(email, password) {
    const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    return res.json();
}

const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');

if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value.trim();
        const password = document.getElementById('loginPassword').value;
        const status = document.getElementById('loginStatus');

        setStatus(status, 'Signing in...');
        const result = await login(email, password);

        if (result.success) {
            localStorage.setItem('zenith_token', result.token);
            localStorage.setItem('zenith_email', email);
            window.location.href = 'dashboard.html';
        } else {
            setStatus(status, result.message || 'Login failed', true);
        }
    });
}

if (signupForm) {
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('signupName').value.trim();
        const email = document.getElementById('signupEmail').value.trim();
        const password = document.getElementById('signupPassword').value;
        const status = document.getElementById('signupStatus');

        setStatus(status, 'Creating account...');
        const result = await signup(name, email, password);

        if (result.success) {
            setStatus(status, 'Account created! Redirecting...');
            setTimeout(() => { window.location.href = 'login.html'; }, 800);
        } else {
            setStatus(status, result.message || 'Signup failed', true);
        }
    });
}
