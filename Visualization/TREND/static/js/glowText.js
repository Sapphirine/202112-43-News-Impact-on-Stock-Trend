let glowInTexts = document.querySelectorAll(".glowIn");
glowInTexts.forEach(glowInText => {
    let letters = glowInText.textContent.split("");
    glowInText.textContent = "";
    letters.forEach((letter, i) => {
        let span = document.createElement("span");
        span.textContent = letter;
        span.style.animationDelay = `${i * 0.05}s`;
        glowInText.append(span);
    });
});

let floatTextMenuLinks = document.querySelectorAll(".float-text-menu li a");
floatTextMenuLinks.forEach(link => {
    let letters = link.textContent.split("");
    link.textContent = "";
    letters.forEach((letter, i) => {
        let span = document.createElement("span");
        span.textContent = letter;
        span.style.transitionDelay = `${i / 20}s`;
        span.dataset.text = letter;
        link.append(span);
    });
});



