const DEBUG = 1;

const email = document.querySelector('.inputEmail');
const errorMsg = document.querySelector('#errorMsg');
const submit = document.querySelector('#button');
const myForm = document.querySelector('.form');
let validEmail = false;

if (DEBUG) console.log(email, myForm);
// myForm.addEventListener("submit", onSubmit);

const goToHomePage = () => {
    document.querySelector('#homePage').style.display = 'block';
    document.querySelector('#confirmPage').style.display = 'none';
    email.value = '';
}

const goToComfirmPage = () => {
    if (validEmail == true) {
        document.querySelector('#homePage').style.display = 'none';
        document.querySelector('#confirmPage').style.display = 'block';
        console.log(`VALID EMAIL ${email.value}`);
        document.querySelector('.msg').innerHTML = `A confirmation email has been sent to <b id="emailValue">${email.value}</b>.  
              Please open it and click the button inside to confirm your subscription.`
    } // if
}; // goToConfirm Page

const showError = () => {
    errorMsg.style.display  = 'block'
    email.style.background = 'hsl(3, 81.80%, 91.40%)';
    email.style.color = 'hsl(4, 100%, 67%)';
    email.style.borderColor = 'hsl(4, 100%, 67%)';
    email.style.outlineColor = 'hsl(4, 100%, 67%)';
}; // showError function

const hideError = () => {
    errorMsg.style.display = 'none';
    email.style.background = '#fff';
    email.style.color = '#000';
    email.style.borderColor = 'hsl(0, 0%,58%);';
    email.style.outlineColor = 'hsl(0, 0%,58%)';
}; // hideError function

submit.addEventListener('click', (e) => {
    e.preventDefault();
    if (!email.value.match(/[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+/)) {
        console.log("error");
        validEmail = false;
        return showError();
    } else {
        console.log("success", email.value);
        validEmail = true;
        goToComfirmPage();
        return hideError();
    } // if 

}); // addEventListener
