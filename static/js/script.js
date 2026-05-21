/* MelodyMind AI - Global Client Scripts */

/**
 * Displays a beautiful DOM-based toast notification.
 * @param {string} message - Message to display
 * @param {string} type - 'info' | 'success' | 'warning' | 'error'
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    // Create elements
    const toast = document.createElement('div');
    toast.className = "flex items-center gap-3 px-4 py-3 rounded-xl border shadow-lg transition-all duration-300 transform translate-y-2 opacity-0 pointer-events-auto max-w-sm";
    
    let bgClass = 'bg-white border-slate-100 text-slate-800';
    let iconHtml = '<i class="fa-solid fa-info text-blue-500"></i>';
    
    if (type === 'success') {
        bgClass = 'bg-emerald-50 border-emerald-100 text-emerald-800';
        iconHtml = '<i class="fa-solid fa-circle-check text-emerald-500"></i>';
    } else if (type === 'error') {
        bgClass = 'bg-red-50 border-red-100 text-red-800';
        iconHtml = '<i class="fa-solid fa-circle-xmark text-red-500"></i>';
    } else if (type === 'warning') {
        bgClass = 'bg-amber-50 border-amber-100 text-amber-800';
        iconHtml = '<i class="fa-solid fa-triangle-exclamation text-amber-500"></i>';
    }
    
    toast.className += ` ${bgClass}`;
    toast.innerHTML = `
        <div class="flex-shrink-0">${iconHtml}</div>
        <div class="text-xs font-semibold flex-grow">${message}</div>
        <button onclick="this.parentElement.remove()" class="text-slate-400 hover:text-slate-700 focus:outline-none">
            <i class="fa-solid fa-xmark text-xs"></i>
        </button>
    `;
    
    container.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.classList.remove('translate-y-2', 'opacity-0');
    }, 10);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.classList.add('translate-y-2', 'opacity-0');
            setTimeout(() => {
                toast.remove();
            }, 300);
        }
    }, 4000);
}
