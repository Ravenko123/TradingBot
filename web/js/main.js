// main.js - landing page scripts

let perfModeEnabled = false;

function enablePerfMode() {
    perfModeEnabled = true;
    document.body.classList.add('perf-mode');
}

function shouldEnablePerfMode() {
    const lowCpu = navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4;
    const lowMem = navigator.deviceMemory && navigator.deviceMemory <= 4;
    return lowCpu || lowMem;
}

function runPerfProbe(done) {
    let frames = 0;
    const start = performance.now();
    function tick(now) {
        frames++;
        if (now - start < 1200) {
            requestAnimationFrame(tick);
        } else {
            const fps = frames / ((now - start) / 1000);
            if (fps < 45) enablePerfMode();
            if (done) done();
        }
    }
    requestAnimationFrame(tick);
}

function checkAuthAndUpdateNav() {
    const token = localStorage.getItem('zenith_token') || localStorage.getItem('ultima_token');
    const navActions = document.getElementById('navActions');
    if (!navActions) return;
    
    if (token) {
        // user is logged in
        navActions.innerHTML = `
            <a href="dashboard.html" class="nav-btn nav-btn-primary"><i class="fas fa-chart-line"></i> Dashboard</a>
            <button id="navLogoutBtn" class="nav-btn nav-btn-ghost" style="border: none; background: rgba(255,255,255,0.04); cursor: pointer;">
                <i class="fas fa-sign-out-alt"></i> Logout
            </button>
        `;
        
        document.getElementById('navLogoutBtn').addEventListener('click', async (e) => {
            e.preventDefault();
            localStorage.removeItem('zenith_token');
            localStorage.removeItem('ultima_token');
            localStorage.removeItem('zenith_email');
            await fetch('http://localhost:5000/api/auth/logout', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            }).catch(() => {});
            window.location.href = 'index.html';
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    checkAuthAndUpdateNav();
    initPageTransitions();
    initSiteReactiveGrid();
    initGridPulseWaves();
    initHeroGlyphInteraction();
    initSectionGlyphAmplifier();
    initDashboardReactiveGrid();
    initParticles();
    initScrollReveal();
    initCounterAnimation();
    initNavScroll();
    initTiltCards();
    initMobileToggle();
});

function initHeroGlyphInteraction() {
    const hero = document.querySelector('.hero');
    const layer = document.querySelector('.bg-glyphs');
    const title = document.querySelector('.hero-title');
    if (!hero || !layer || !title) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let glyphs = [...layer.querySelectorAll('.bg-glyph')];
    if (!glyphs.length) return;

    const heroKeep = Math.min(14, glyphs.length);
    glyphs.slice(heroKeep).forEach((g) => g.remove());
    glyphs = glyphs.slice(0, heroKeep);

    const glyphMeta = glyphs.map((glyph, index) => ({
        glyph,
        baseX: 0,
        baseY: 0,
        react: Number(glyph.style.getPropertyValue('--react')) || 0.26,
        amp: 8 + (index % 5) * 2,
        speed: 0.7 + (index % 7) * 0.09,
        phase: Math.random() * Math.PI * 2,
    }));

    let pointerTargetX = 0.5;
    let pointerTargetY = 0.42;
    let pointerX = pointerTargetX;
    let pointerY = pointerTargetY;
    let pointerActive = false;

    const cacheBase = () => {
        const heroRect = hero.getBoundingClientRect();
        glyphMeta.forEach((item) => {
            const rect = item.glyph.getBoundingClientRect();
            item.baseX = (rect.left + rect.width * 0.5) - heroRect.left;
            item.baseY = (rect.top + rect.height * 0.5) - heroRect.top;
        });
    };

    const updatePointer = (clientX, clientY) => {
        const rect = hero.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        pointerTargetX = (clientX - rect.left) / rect.width;
        pointerTargetY = (clientY - rect.top) / rect.height;
    };

    const render = (time) => {
        pointerX += (pointerTargetX - pointerX) * 0.1;
        pointerY += (pointerTargetY - pointerY) * 0.1;

        hero.style.setProperty('--mx', `${(pointerX * 100).toFixed(2)}%`);
        hero.style.setProperty('--my', `${(pointerY * 100).toFixed(2)}%`);
        hero.style.setProperty('--mxn', ((pointerX - 0.5) * 2).toFixed(4));
        hero.style.setProperty('--myn', ((pointerY - 0.5) * 2).toFixed(4));
        layer.style.setProperty('--hover', pointerActive ? '0.18' : '0.08');

        const heroRect = hero.getBoundingClientRect();
        const mouseX = pointerX * heroRect.width;
        const mouseY = pointerY * heroRect.height;

        glyphMeta.forEach((item) => {
            const swayX = Math.sin(time * 0.00055 * item.speed + item.phase) * item.amp;
            const swayY = Math.cos(time * 0.0005 * item.speed + item.phase * 1.17) * (item.amp * 0.72);

            const dx = mouseX - item.baseX;
            const dy = mouseY - item.baseY;
            const dist = Math.hypot(dx, dy);
            const influence = Math.max(0, 1 - dist / 360);

            const mousePushX = -dx * item.react * influence * 0.16;
            const mousePushY = -dy * item.react * influence * 0.1;

            item.glyph.style.setProperty('--pull-x', `${(swayX + mousePushX).toFixed(2)}px`);
            item.glyph.style.setProperty('--pull-y', `${(swayY + mousePushY).toFixed(2)}px`);
        });

        requestAnimationFrame(render);
    };

    hero.addEventListener('mousemove', (event) => {
        pointerActive = true;
        updatePointer(event.clientX, event.clientY);
    }, { passive: true });

    hero.addEventListener('mouseenter', (event) => {
        pointerActive = true;
        updatePointer(event.clientX, event.clientY);
    }, { passive: true });

    hero.addEventListener('mouseleave', () => {
        pointerActive = false;
        pointerTargetX = 0.5;
        pointerTargetY = 0.42;
    });

    window.addEventListener('resize', cacheBase);
    cacheBase();
    requestAnimationFrame(render);
}

function initSectionGlyphAmplifier() {
    if (perfModeEnabled) return;
    const sections = [...document.querySelectorAll('.section-glyphs')];
    if (!sections.length) return;

    const sectionGlyphMeta = [];

    sections.forEach((container) => {
        let glyphs = [...container.querySelectorAll('.bg-glyph')];
        if (!glyphs.length) return;

        const keep = Math.min(6, glyphs.length);
        glyphs.slice(keep).forEach((g) => g.remove());
        glyphs = glyphs.slice(0, keep);

        const items = glyphs.map((glyph, idx) => ({
            glyph,
            baseX: 0,
            baseY: 0,
            react: 0.1 + (idx % 4) * 0.02,
            driftAmp: 3 + (idx % 3),
            speed: 0.75 + (idx % 5) * 0.07,
            phase: Math.random() * Math.PI * 2,
        }));

        sectionGlyphMeta.push({ container, items });
    });

    if (!sectionGlyphMeta.length) return;

    const pointer = {
        x: window.innerWidth * 0.5,
        y: window.innerHeight * 0.5,
        tx: window.innerWidth * 0.5,
        ty: window.innerHeight * 0.5,
    };

    const cacheBase = () => {
        sectionGlyphMeta.forEach((group) => {
            const rect = group.container.getBoundingClientRect();
            group.items.forEach((item) => {
                const glyphRect = item.glyph.getBoundingClientRect();
                item.baseX = (glyphRect.left + glyphRect.width * 0.5) - rect.left;
                item.baseY = (glyphRect.top + glyphRect.height * 0.5) - rect.top;
            });
        });
    };

    const render = (time) => {
        pointer.x += (pointer.tx - pointer.x) * 0.09;
        pointer.y += (pointer.ty - pointer.y) * 0.09;

        sectionGlyphMeta.forEach((group) => {
            const rect = group.container.getBoundingClientRect();
            const localX = pointer.x - rect.left;
            const localY = pointer.y - rect.top;

            group.items.forEach((item) => {
                const dx = localX - item.baseX;
                const dy = localY - item.baseY;
                const dist = Math.hypot(dx, dy);
                const influence = Math.max(0, 1 - dist / 320);

                const driftX = Math.sin(time * 0.0005 * item.speed + item.phase) * item.driftAmp;
                const driftY = Math.cos(time * 0.00045 * item.speed + item.phase * 1.2) * (item.driftAmp * 0.65);
                const pushX = -dx * item.react * influence * 0.14;
                const pushY = -dy * item.react * influence * 0.1;

                item.glyph.style.setProperty('--pull-x', `${(driftX + pushX).toFixed(2)}px`);
                item.glyph.style.setProperty('--pull-y', `${(driftY + pushY).toFixed(2)}px`);
            });
        });

        requestAnimationFrame(render);
    };

    window.addEventListener('mousemove', (event) => {
        pointer.tx = event.clientX;
        pointer.ty = event.clientY;
    }, { passive: true });

    window.addEventListener('resize', cacheBase);
    cacheBase();
    requestAnimationFrame(render);
}

function initSiteReactiveGrid() {
    const grids = [...document.querySelectorAll('.site-reactive-grid, #dashReactiveGrid')];
    if (!grids.length) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const hero = document.querySelector('.hero');
    const isDashboard = document.body.classList.contains('dashboard-body');
    const idleHover = isDashboard ? 0.72 : 0.58;
    const activeHover = isDashboard ? 1.46 : 1.12;

    let targetX = window.innerWidth * 0.5;
    let targetY = window.innerHeight * 0.5;
    let currentX = targetX;
    let currentY = targetY;
    let hoverTarget = idleHover;
    let hoverCurrent = idleHover;
    let rafId = null;

    const render = () => {
        currentX += (targetX - currentX) * 0.115;
        currentY += (targetY - currentY) * 0.115;
        hoverCurrent += (hoverTarget - hoverCurrent) * 0.11;

        const nx = (currentX / window.innerWidth - 0.5) * 2;
        const ny = (currentY / window.innerHeight - 0.5) * 2;
        grids.forEach((grid) => {
            grid.style.setProperty('--mx', `${((nx * 0.5 + 0.5) * 100).toFixed(2)}%`);
            grid.style.setProperty('--my', `${((ny * 0.5 + 0.5) * 100).toFixed(2)}%`);
            grid.style.setProperty('--mxn', '0');
            grid.style.setProperty('--myn', '0');
            grid.style.setProperty('--grid-hover', hoverCurrent.toFixed(4));
        });

        if (Math.abs(targetX - currentX) > 0.04 || Math.abs(targetY - currentY) > 0.04 || Math.abs(hoverTarget - hoverCurrent) > 0.001) {
            rafId = requestAnimationFrame(render);
        } else {
            rafId = null;
        }
    };

    const schedule = () => {
        if (!rafId) rafId = requestAnimationFrame(render);
    };

    window.addEventListener('mousemove', (event) => {
        targetX = event.clientX;
        targetY = event.clientY;

        const heroBottom = hero ? hero.getBoundingClientRect().bottom : 0;
        const inTopHero = event.clientY <= heroBottom;
        hoverTarget = inTopHero ? idleHover : activeHover;

        if (inTopHero) {
            targetX = window.innerWidth * 0.5;
            targetY = window.innerHeight * 0.5;
        }
        schedule();
    }, { passive: true });

    window.addEventListener('mouseleave', () => {
        hoverTarget = idleHover;
        targetX = window.innerWidth * 0.5;
        targetY = window.innerHeight * 0.5;
        schedule();
    });

    window.addEventListener('resize', () => {
        targetX = window.innerWidth * 0.5;
        targetY = window.innerHeight * 0.5;
        schedule();
    });

    schedule();
}

function initDashboardReactiveGrid() {
    return;
}

function initGridSnakes() {
    if (perfModeEnabled) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const grid = document.querySelector('.site-reactive-grid');
    if (!grid) return;

    const CELL = 76;
    const MAX_ACTIVE = 8;
    const LIFE_MS = 18500;

    const DIRS = [
        { x: 1, y: 0 },
        { x: -1, y: 0 },
        { x: 0, y: 1 },
        { x: 0, y: -1 }
    ];

    const cellKey = (x, y) => `${x}:${y}`;

    const perpendicularTurns = (dir) => {
        if (dir.x !== 0) {
            return [{ x: 0, y: 1 }, { x: 0, y: -1 }];
        }
        return [{ x: 1, y: 0 }, { x: -1, y: 0 }];
    };

    const buildSnakePath = (cols, rows) => {
        const stepsTarget = 220 + Math.floor(Math.random() * 180);
        let x = Math.floor(Math.random() * cols);
        let y = Math.floor(Math.random() * rows);
        let dir = DIRS[Math.floor(Math.random() * DIRS.length)];

        const points = [{ x, y }];
        const pointSet = new Set([cellKey(x, y)]);

        for (let step = 1; step < stepsTarget; step++) {
            let candidateDirs = [dir];

            // Every second grid cell: 50/50 turn right/left (perpendicular)
            if (step % 2 === 0) {
                const turns = perpendicularTurns(dir);
                candidateDirs = Math.random() < 0.5 ? [turns[0], turns[1], dir] : [turns[1], turns[0], dir];
            }

            // Fallback options if blocked
            const fallback = DIRS
                .filter((d) => !(d.x === -dir.x && d.y === -dir.y))
                .sort(() => Math.random() - 0.5);

            const allCandidates = [...candidateDirs, ...fallback];
            let moved = false;

            for (const nextDir of allCandidates) {
                const nx = x + nextDir.x;
                const ny = y + nextDir.y;
                if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
                const key = cellKey(nx, ny);
                if (pointSet.has(key)) continue;

                x = nx;
                y = ny;
                dir = nextDir;
                points.push({ x, y });
                pointSet.add(key);
                moved = true;
                break;
            }

            if (!moved) break;
        }

        if (points.length < 7) return null;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        points.forEach((p) => {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        });

        const localPoints = points.map((p) => ({ x: p.x - minX, y: p.y - minY }));
        const d = localPoints
            .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x * CELL + 0.5} ${p.y * CELL + 0.5}`)
            .join(' ');

        return {
            globalPoints: points,
            d,
            minX,
            minY,
            w: maxX - minX + 1,
            h: maxY - minY + 1,
            len: (points.length - 1) * CELL
        };
    };

    const occupied = new Set();
    let tick = null;

    const spawnOne = () => {
        const cols = Math.max(8, Math.floor(window.innerWidth / CELL));
        const rows = Math.max(8, Math.floor(window.innerHeight / CELL));
        if (!cols || !rows) return;

        for (let attempt = 0; attempt < 36; attempt++) {
            const snakePath = buildSnakePath(cols, rows);
            if (!snakePath) continue;

            // Check exact walked path cells are free
            let conflict = false;
            const keys = snakePath.globalPoints.map((p) => cellKey(p.x, p.y));
            for (const key of keys) {
                if (occupied.has(key)) {
                    conflict = true;
                    break;
                }
            }
            if (conflict) continue;

            keys.forEach(k => occupied.add(k));

            const pxW = snakePath.w * CELL;
            const pxH = snakePath.h * CELL;

            const snake = document.createElement('div');
            snake.className = 'grid-snake';
            snake.style.left = `${snakePath.minX * CELL}px`;
            snake.style.top = `${snakePath.minY * CELL}px`;
            snake.style.width = `${pxW}px`;
            snake.style.height = `${pxH}px`;

            const reverse = Math.random() < 0.5;
            const headLen = 22 + Math.floor(Math.random() * 14);
            const dur = 10.8 + Math.random() * 4.6;

            snake.innerHTML = `
                <svg class="grid-snake-svg" viewBox="0 0 ${pxW} ${pxH}" preserveAspectRatio="none" aria-hidden="true">
                    <path class="grid-snake-head" d="${snakePath.d}" />
                </svg>
            `;

            grid.appendChild(snake);

            // Animate with Web Animations API for variable-length paths
            const headEl = snake.querySelector('.grid-snake-head');
            const startOff = reverse ? -snakePath.len : 0;
            const endOff = reverse ? 0 : -snakePath.len;

            headEl.animate([
                { strokeDashoffset: `${startOff}`, strokeDasharray: `${headLen} 999`, opacity: 0.92, strokeWidth: '2.6' },
                { strokeDashoffset: `${endOff}`, strokeDasharray: `${Math.max(10, headLen * 0.78)} 999`, opacity: 0.84, strokeWidth: '2.3', offset: 0.85 },
                { strokeDashoffset: `${endOff}`, strokeDasharray: '0.1 999', opacity: 0, strokeWidth: '0.5' }
            ], {
                duration: dur * 1000,
                easing: 'cubic-bezier(0.16, 0.78, 0.24, 1)',
                fill: 'forwards'
            });

            const life = LIFE_MS + Math.random() * 4200;
            setTimeout(() => {
                snake.remove();
                keys.forEach(k => occupied.delete(k));
            }, life);
            return;
        }
    };

    const reset = () => {
        grid.querySelectorAll('.grid-snake').forEach((el) => el.remove());
        occupied.clear();
    };

    let resizeTimer = null;
    window.addEventListener('resize', () => {
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(reset, 140);
    });

    tick = setInterval(() => {
        if (document.hidden) return;
        if (occupied.size < MAX_ACTIVE) spawnOne();
    }, 420);

    for (let i = 0; i < 5; i++) spawnOne();
}

function initGridPulseWaves() {
    if (perfModeEnabled) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const grid = document.querySelector('.site-reactive-grid');
    if (!grid) return;

    let pulseLayer = grid.querySelector('.grid-pulse-layer');
    if (!pulseLayer) {
        pulseLayer = document.createElement('div');
        pulseLayer.className = 'grid-pulse-layer';
        grid.appendChild(pulseLayer);
    }

    const spawnPulse = () => {
        if (document.hidden) return;

        const wave = document.createElement('span');
        wave.className = 'grid-pulse-wave';

        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const size = 180 + Math.random() * 360;
        const dur = 3200 + Math.random() * 2600;
        const delay = Math.random() * 260;

        wave.style.left = `${x}%`;
        wave.style.top = `${y}%`;
        wave.style.setProperty('--pulse-size', `${size}px`);
        wave.style.animationDuration = `${dur}ms`;
        wave.style.animationDelay = `${delay}ms`;

        pulseLayer.appendChild(wave);
        setTimeout(() => wave.remove(), dur + delay + 80);
    };

    for (let i = 0; i < 4; i++) setTimeout(spawnPulse, i * 240);
    setInterval(spawnPulse, 520);
}

// background — full-page falling stars (not viewport-locked)
function initParticles() {
    if (perfModeEnabled) return;
    const fullCanvas = document.getElementById('particleCanvas');
    const planetCanvas = document.getElementById('planetParticleCanvas');
    const fields = [];

    if (fullCanvas) {
        fields.push({
            canvas: fullCanvas,
            ctx: fullCanvas.getContext('2d'),
            stars: [],
            count: 19,
            speed: 1,
            alphaBoost: 1
        });
    }

    if (planetCanvas) {
        fields.push({
            canvas: planetCanvas,
            ctx: planetCanvas.getContext('2d'),
            stars: [],
            count: 16,
            speed: 0.92,
            alphaBoost: 1.1
        });
    }

    if (!fields.length) return;

    const measureField = (field) => {
        const { canvas } = field;
        if (canvas.id === 'particleCanvas') {
            field.w = window.innerWidth;
            field.h = window.innerHeight;
            canvas.style.width = `${field.w}px`;
            canvas.style.height = `${field.h}px`;
        } else {
            field.w = Math.max(1, Math.floor(canvas.clientWidth));
            field.h = Math.max(1, Math.floor(canvas.clientHeight));
        }

        canvas.width = field.w;
        canvas.height = field.h;
    };

    fields.forEach(measureField);

    let measureRaf = 0;
    const scheduleMeasure = () => {
        if (measureRaf) return;
        measureRaf = requestAnimationFrame(() => {
            measureRaf = 0;
            fields.forEach(measureField);
        });
    };
    window.addEventListener('resize', scheduleMeasure);

    const seedStars = (field) => {
        field.stars.length = 0;
        for (let i = 0; i < field.count; i++) {
            field.stars.push({
                x: Math.random() * field.w,
                y: Math.random() * field.h,
                len: Math.random() * 24 + 14,
                r: Math.random() * 2.1 + 1.4,
                alpha: Math.random() * 0.42 + 0.34,
                twinkle: Math.random() * 0.6 + 0.5,
                phase: Math.random() * Math.PI * 2,
                green: Math.random() < 0.2,
                vy: (Math.random() * 1.1 + 0.45) * field.speed,
                vx: -(Math.random() * 1.0 + 0.35) * field.speed
            });
        }
    };

    fields.forEach(seedStars);

    let lastFrame = 0;
    function animate(time) {
        if (perfModeEnabled) return;
        if (time - lastFrame < 36) { requestAnimationFrame(animate); return; }
        lastFrame = time;
        fields.forEach((field) => {
            const { ctx, stars, w, h, alphaBoost } = field;
            ctx.clearRect(0, 0, w, h);

            stars.forEach((star) => {
                const pulse = 0.7 + 0.3 * Math.sin(time * 0.0012 * star.twinkle + star.phase);
                const a = star.alpha * pulse * alphaBoost;
                const c = star.green ? '0,255,135' : '255,255,255';

                star.y += star.vy;
                star.x += star.vx;

                if (star.y > h + 24 || star.x < -40) {
                    if (Math.random() < 0.62) {
                        star.x = w + Math.random() * 40;
                        star.y = Math.random() * h * 0.42;
                    } else {
                        star.x = Math.random() * w;
                        star.y = -40;
                    }
                }

                const tailX = star.x - star.vx * star.len;
                const tailY = star.y - star.vy * star.len;

                const grad = ctx.createLinearGradient(star.x, star.y, tailX, tailY);
                grad.addColorStop(0, `rgba(${c},${Math.min(1, a * 1.15).toFixed(3)})`);
                grad.addColorStop(1, `rgba(${c},0)`);

                ctx.beginPath();
                ctx.moveTo(star.x, star.y);
                ctx.lineTo(tailX, tailY);
                ctx.strokeStyle = grad;
                ctx.lineWidth = star.green ? 2.2 : 1.6;
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${c},${Math.min(1, a * 1.35).toFixed(3)})`;
                ctx.shadowBlur = star.green ? 28 : 22;
                ctx.shadowColor = `rgba(${c},${Math.min(1, a * 1.1).toFixed(3)})`;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
        });

        requestAnimationFrame(animate);
    }
    animate(0);
}

// scroll reveal
function initScrollReveal() {
    const els = document.querySelectorAll('.reveal-up, .reveal-left, .reveal-right, .reveal-scale');
    if (!els.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                // Stagger for siblings
                const siblings = entry.target.parentElement.querySelectorAll('.reveal-up, .reveal-left, .reveal-right, .reveal-scale');
                let delay = 0;
                siblings.forEach((sib, idx) => {
                    if (sib === entry.target) delay = idx * 80;
                });
                setTimeout(() => entry.target.classList.add('revealed'), delay);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

    els.forEach(el => observer.observe(el));
}

// counters
function initCounterAnimation() {
    const counters = document.querySelectorAll('[data-count]');
    if (!counters.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    counters.forEach(c => observer.observe(c));
}

function animateCounter(el) {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || '';
    const prefix = el.dataset.prefix || '';
    const duration = 2200;
    let start = 0;
    const startTime = performance.now();

    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out quart
        const ease = 1 - Math.pow(1 - progress, 4);
        const current = Math.floor(ease * target);

        el.textContent = prefix + current.toLocaleString() + suffix;

        if (progress < 1) requestAnimationFrame(update);
        else el.textContent = prefix + target.toLocaleString() + suffix;
    }

    requestAnimationFrame(update);
}

// nav scroll
function initNavScroll() {
    const nav = document.querySelector('.navbar');
    if (!nav) return;

    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const s = window.scrollY;
        if (s > 50) nav.classList.add('scrolled');
        else nav.classList.remove('scrolled');
        lastScroll = s;
    }, { passive: true });

    // Active section highlighting
    const links = document.querySelectorAll('.nav-link[href^="#"]');
    const sections = [];
    links.forEach(link => {
        const id = link.getAttribute('href').slice(1);
        const section = document.getElementById(id);
        if (section) sections.push({ el: section, link });
    });

    if (sections.length) {
        const sectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    links.forEach(l => l.classList.remove('active'));
                    const match = sections.find(s => s.el === entry.target);
                    if (match) match.link.classList.add('active');
                }
            });
        }, { rootMargin: '-40% 0px -55% 0px' });

        sections.forEach(s => sectionObserver.observe(s.el));
    }
}

// tilt cards
function initTiltCards() {
    if (document.body.classList.contains('dashboard-body')) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (window.innerWidth < 992) return;

    const cards = document.querySelectorAll('[data-tilt], .pricing-section .pricing-card, .performance-section .perf-card, .strategy-section .timeline-card, .prop-why-section .prop-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = (y - centerY) / centerY * -6;
            const rotateY = (x - centerX) / centerX * 6;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02,1.02,1.02)`;
            card.style.transition = 'transform 0.1s ease';

            const px = (x / rect.width) * 100;
            const py = (y / rect.height) * 100;
            card.style.setProperty('--mouse-x', px + '%');
            card.style.setProperty('--mouse-y', py + '%');

            // Update glow position
            const glowEl = card.querySelector('.feature-card-glow');
            if (glowEl) {
                glowEl.style.opacity = '1';
            }
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1,1,1)';
            card.style.transition = 'transform 0.5s ease';
        });
    });
}

// mobile nav
function initMobileToggle() {
    const toggle = document.querySelector('.mobile-toggle');
    const navLinks = document.querySelector('.nav-links');
    if (!toggle || !navLinks) return;

    toggle.addEventListener('click', () => {
        navLinks.classList.toggle('open');
    });

    // Close on link click
    navLinks.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => navLinks.classList.remove('open'));
    });
}

function initPageTransitions() {
    const overlay = document.getElementById('pageTransition');
    if (!overlay) return;
    // Fade overlay out once page loads (handles forward nav)
    overlay.classList.remove('going-out');

    document.querySelectorAll('a[href]').forEach(link => {
        const href = link.getAttribute('href');
        if (!href) return;
        // Skip: pure anchors, external links, mailto, js
        if (href.startsWith('#') || href.startsWith('http') || href.startsWith('//') ||
            href.startsWith('mailto:') || href.startsWith('javascript')) return;
        // Skip hash-only navigation on same page anchors within index
        if (href.includes('#') && !href.includes('.html')) return;

        link.addEventListener('click', e => {
            e.preventDefault();
            const dest = href;
            overlay.classList.add('going-out');
            setTimeout(() => { window.location.href = dest; }, 420);
        });
    });
}
