const DEBUG = 0;

let ccInformation = []

const ccInputPage = document.querySelector("#card-info");
const thankYouPage = document.querySelector("#thank-you-page");

const ccName = document.querySelector("#name-text");
const ccNumber = document.querySelector("#card-number")
const ccMonth = document.querySelector("#card-month");
const ccYear = document.querySelector("#card-year");
const ccCvc = document.querySelector("#cvc-number");


const input = document.querySelectorAll("input");

function padZero(number) {
    if (DEBUG) console.log("HERE");
    if (number == 0) return "00";
    if (number < 10 ? trimmed = "0" + number.replace(/^0+/, "") : trimmed = number);
    return trimmed;
}


function displayccName(e) {
    if (DEBUG) console.log("updating");
    if (DEBUG) console.log(ccName.value, ccNumber.value, ccMonth.value, ccYear.value, ccCvc.value);
    document.querySelector("#front-card-name").innerHTML = `${ccName.value}`
}

function displayccNumber(e) {
    if (ccNumber.value.length <= 16) {
        let chunks = ccNumber.value.match(/.{1,4}/g || []);
        let formattedValue = chunks.join(' ');
        document.querySelector("#front-info-numbers").innerHTML = `${formattedValue}`
        console.log("Valid card number");
    } else {
        console.log("Must be a valid card number");
    }
    
}

function displayccExpiry(e) {
    if ((ccMonth.value <= 12 && ccMonth.value > 0) && ccYear.value.length <= 2) {
        console.log("valid expiry date");
        document.querySelector("#front-card-expiry").innerHTML = `${padZero(ccMonth.value)}/${padZero(ccYear.value)}`;
        
    } else {
        console.log("Must be a valid expiry date");
    }
    
}

function displayccCVC(e) {
    if (ccCvc.value.length <= 3) {
        document.querySelector("#back-info").innerHTML = `${ccCvc.value}`
        console.log("Valid CVC number");
    } else {
        console.log("Must be a valid CVC number");
    }
}

function confirm() {
    thankYouPage.style.display = "grid";
    ccInputPage.style.display = "none";
}

function setDefault() {
    document.querySelector("#front-card-name").innerHTML = "Jane Appleseed";
    document.querySelector("#front-info-numbers").innerHTML = "0000 0000 0000 0000";
    document.querySelector("#front-card-expiry").innerHTML = "00/00";
    document.querySelector("#back-info").innerHTML = "000"
}

function displayInputPage() {
    thankYouPage.style.display = "none";
    ccInputPage.style.display = "grid";
    ccName.value = "";
    ccNumber.value = "";
    ccMonth.value = "";
    ccYear.value = "";
    ccCvc.value = "";

    setDefault();
}

const myForm = addEventListener("submit", onSubmit);

function onSubmit(e) {
    e.preventDefault();
    
    const dataObject = {
        "name" : ccName.value, 
        "number" : ccNumber.value, 
        "month" : ccMonth.value, 
        "year" : ccYear.value, 
        "cvc" : ccCvc.value
    };

    fetch('/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dataObject),
    })
        .then(res => {
            if (!res.ok) throw new Error('Network response was not ok');
            return res.json();
        })
        .then(json => {
            console.log('Saved successfully:', json);
        })
        .catch(err => {
            console.error('Error saving data:', err);
        });
    

    console.log(dataObject);
    
    ccInformation.push(dataObject);
    console.log(ccInformation);

    confirm();

}