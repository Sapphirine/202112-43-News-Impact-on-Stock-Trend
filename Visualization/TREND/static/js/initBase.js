let duration = 0;
let delay = 0;
let revealText = document.querySelector(".reveal");
let letters = revealText.textContent.split("");
revealText.textContent = "";
let middle = letters.filter(e => e !== " ").length / 2;
letters.forEach((letter, i) => {
    let span = document.createElement("span");
    span.textContent = letter;
    span.style.animationDelay = `${delay + Math.abs(i - middle) * 0.1}s`;
    revealText.append(span);
});
var searchbox = document.getElementById("searchBox");
setTimeout(function() { searchbox.style.display = ""; }, 1000);