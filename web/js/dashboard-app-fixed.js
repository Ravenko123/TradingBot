// sidebar nav - CLEAN WORKING VERSION
function initSidebarNav() {
    const navLinks = document.querySelectorAll('.sidebar-nav a[data-section]');
    const sections = document.querySelectorAll('.dashboard-main > section');
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.getElementById('mobileNavToggle');
    
    console.log('✓ Nav init - links:', navLinks.length, 'sections:', sections.length);

    function switchToSection(sectionId) {
        console.log('→ Switching to:', sectionId);
        
        // Hide all sections first
        sections.forEach(sec => {
            sec.style.display = 'none';
            sec.classList.remove('active');
        });

        // Show target section
        const target = document.getElementById(sectionId);
        if (target) {
            target.style.display = 'flex';
            target.classList.add('active');
        } else {
            console.error('✗ Section not found:', sectionId);
        }

        // Update nav links
        navLinks.forEach(link => link.classList.remove('active'));
        const activeLink = document.querySelector(`.sidebar-nav a[data-section="${sectionId}"]`);
        if (activeLink) activeLink.classList.add('active');

        // Close mobile menu
        if (sidebar && toggleBtn) {
            sidebar.classList.remove('active');
            toggleBtn.classList.remove('active');
        }
    }

    // Attach click handlers
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = link.getAttribute('data-section');
            switchToSection(sectionId);
        });
    });

    // Mobile toggle
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('active');
            toggleBtn.classList.toggle('active');
        });
    }

    // Initialize with overview
    sections.forEach(sec => sec.style.display = 'none');
    const overview = document.getElementById('overview');
    if (overview) {
        overview.style.display = 'flex';
        overview.classList.add('active');
    }
    if (navLinks[0]) navLinks[0].classList.add('active');
}
