const DEBUG = 0;

let ccInformation = []

let validNumber;
let validExpiry;
let validCvc;

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
        validNumber = true;
        console.log("Valid card number");
    } else {
        validNumber = false;
        console.log("Must be a valid card number");
    }
    
}

function displayccExpiry(e) {
    if ((ccMonth.value <= 12 && ccMonth.value > 0) && ccYear.value.length <= 2) {
        document.querySelector("#front-card-expiry").innerHTML = `${padZero(ccMonth.value)}/${padZero(ccYear.value)}`;
        validExpiry = true;
        console.log("valid expiry date");
    } else {
        validExpiry = false;
        console.log("Must be a valid expiry date");
    }
    
}

function displayccCVC(e) {
    if (ccCvc.value.length <= 3) {
        document.querySelector("#back-info").innerHTML = `${ccCvc.value}`
        validCvc = true;
        console.log("Valid CVC number");
    } else {
        validCvc = false;
        console.log("Must be a valid CVC number");
    }
}

async function confirm() {

    const formData = new FormData(ccInputPage);

    const name = formData.get('name');
    const number = formData.get('number');
    const month = padZero(formData.get('month'));
    const year = padZero(formData.get('year'));
    const cvc = formData.get('cvc');

    if (DEBUG) console.log("HELLO WORLD THIS WORKED", name, number, month, year, cvc);

    try {
        const res = await fetch("http://localhost:5000/user", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, number, month, year, cvc })
        });

        if (!res.ok) {
            throw new Error('Failed to send cc information');
        }
    } catch (error) {
        console.log('Error creating entry', error);
    }

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

function downloadJSON(obj, filename) {
    // turn object into JSON string
    const jsonString = JSON.stringify(obj, null, 2);

    // create a blob containing json of mime-type application/json
    const blob = new Blob([jsonString], { type:'application/json' });

    // create an object URL from that block
    const url = URL.createObjectURL(blob);

    // Create a temp <a> tag w/ download attribute
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;

    // Programmatically click <a> to open the "Save As" dialogue
    document.body.appendChild(a);
    a.click();

    // Cleanup: revoke object url and remove <a> tag
    setTimeout(() => {
        URL.revokeObjectURL(url);
        a.remove();
    }, 0)
}

const myForm = addEventListener("submit", onSubmit);

function onSubmit(e) {
    e.preventDefault();

    // downloadJSON(dataObject, 'data/cc.json');

    // fetch('/save', {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify(dataObject),
    // })
    //     .then(res => {
    //         if (!res.ok) throw new Error('Network response was not ok');
    //         return res.json();
    //     })
    //     .then(json => {
    //         console.log('Saved successfully:', json);
    //     })
    //     .catch(err => {
    //         console.error('Error saving data:', err);
    //     });
    
    if (DEBUG) console.log(ccInformation);
    if (DEBUG) console.log(validNumber, validExpiry, validCvc)
    if (validNumber && validExpiry && validCvc) { 
        confirm();
        console.log("Thank you, recorded!");
    } else {
        console.log("Must enter valid inputs");
    }

}