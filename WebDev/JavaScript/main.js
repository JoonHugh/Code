// DOM practice
// changing things in the DOM == changing the user interface

console.log(window);

/*
// Single element selectors
const form = document.getElementById("my-form");
console.log(form)

console.log(document.querySelector(".container"));
console.log(document.querySelector("h1"));


// Multi element selectors

console.log(document.querySelectorAll(".item"));

const items = document.querySelectorAll(".item");
// items.forEach((item) => console.log(item));

const ul = document.querySelector(".items");
// ul.remove();
// ul.lastElementChild.remove();
// ul.firstElementChild.textContent = "Hello";
// ul.children[1].innerText = "brad";
//ul.lastElementChild.innerHTML = "<h1>Hello World</h1>";

const btn = document.querySelector((".btn"));

console.log(btn);
btn.style.background = "red";
// console.log(ul);

// getting a button to do something
btn.addEventListener("click", (e) => { // first parameter = what you're doing, second = event object => function
    e.preventDefault();
    // console.log(e.target.);
    document.querySelector("#my-form").style.background = "#ccc";
    document.querySelector("body").classList.add("bg-dark");
    ul.lastElementChild.innerHTML = "<h1>Hello</h1>"


});
*/


const myForm = document.querySelector("#my-form");
const nameInput = document.querySelector("#name");
const emailInput = document.querySelector("#email");
const msg = document.querySelector(".msg");
const userList = document.querySelector('#users');

myForm.addEventListener("submit", onSubmit)

function onSubmit(e) {
    e.preventDefault();

    if (nameInput.value  === "" || emailInput.value === "") {
        msg.classList.add("error");
        msg.innerHTML = "Please enter both fields";

        setTimeout(() => msg.remove(), 3000);
    } else {
        // console.log("Success");
        const li = document.createElement('li');
        li.appendChild(document.createTextNode(`${nameInput.value} : ${emailInput.value}`));

        userList.appendChild(li);

        // Clear fields
        nameInput.value = '';
        emailInput.value = '';
    }

    console.log(nameInput.value);
    console.log(emailInput.value);

}