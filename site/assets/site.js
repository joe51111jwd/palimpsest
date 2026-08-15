/* Palimpsest — site behaviour.
   Two things only: theme persistence and the mobile nav toggle.
   No dependencies, no network, no build step.

   Load it in <head> WITHOUT defer:
       <script src="assets/site.js"></script>
   The theme is applied during head parsing so there is no flash of the wrong
   theme; everything else waits for DOMContentLoaded. */

(function () {
  "use strict";

  var KEY = "palimpsest-theme";

  function read() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }

  function write(value) {
    try { localStorage.setItem(KEY, value); } catch (e) { /* file:// or blocked */ }
  }

  function apply(theme) {
    var root = document.documentElement;
    if (theme === "dark" || theme === "light") {
      root.setAttribute("data-theme", theme);
    } else {
      root.removeAttribute("data-theme");
    }
  }

  function systemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark" : "light";
  }

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || systemTheme();
  }

  /* Run immediately: before first paint. */
  apply(read());

  function labelToggle(button) {
    var next = currentTheme() === "dark" ? "light" : "dark";
    button.textContent = next === "dark" ? "Dark" : "Light";
    button.setAttribute("aria-label", "Switch to " + next + " theme");
  }

  function initTheme() {
    var button = document.querySelector("[data-theme-toggle]");
    if (!button) return;
    labelToggle(button);
    button.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      apply(next);
      write(next);
      labelToggle(button);
    });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var onChange = function () { if (!read()) labelToggle(button); };
      if (mq.addEventListener) mq.addEventListener("change", onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }
  }

  function initNav() {
    var header = document.querySelector(".site-header");
    var button = document.querySelector("[data-nav-toggle]");
    if (!header || !button) return;

    function set(open) {
      header.setAttribute("data-nav", open ? "open" : "closed");
      button.setAttribute("aria-expanded", open ? "true" : "false");
    }

    set(false);
    button.addEventListener("click", function () {
      set(header.getAttribute("data-nav") !== "open");
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && header.getAttribute("data-nav") === "open") {
        set(false);
        button.focus();
      }
    });
  }

  /* Any .table-wrap that actually overflows becomes keyboard-scrollable and
     is announced as a scrollable region. */
  function initTables() {
    var wraps = document.querySelectorAll(".table-wrap");
    for (var i = 0; i < wraps.length; i++) {
      var wrap = wraps[i];
      if (wrap.scrollWidth > wrap.clientWidth && !wrap.hasAttribute("tabindex")) {
        wrap.setAttribute("tabindex", "0");
        wrap.setAttribute("role", "region");
      }
    }
  }

  function init() {
    initTheme();
    initNav();
    initTables();
    window.addEventListener("resize", initTables);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
