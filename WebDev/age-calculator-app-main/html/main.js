const DEBUG = 0;

let validYear;

const submit = document.querySelector('#submit')


const form = document.querySelector('.form')
console.log(form)
console.log("Hello World")

const date = new Date();

console.log(date.getFullYear(), date.getMonth(), date.getDate());
if (DEBUG) console.log(typeof date.getFullYear());

function invalidYear() {
    console.log("INVALID YEAR");
    validDate = false;
}

function invalidMonth() {
    console.log("INVALID MONTH");
    validDate = false;
}

function invalidDay() {
    console.log("INVALID DAY");
    validDate = false;
}


submit.addEventListener('click', (e) => {
    if (DEBUG) console.log("CLICKED");

    validDate = true;

    const year = document.querySelector('#inputYear');
    const month = document.querySelector('#inputMonth');
    const day = document.querySelector('#inputDay');

    if (year.value.trim() == "") {
        invalidYear();
    }
    if (month.value.trim() == "") {
        invalidMonth();
    }
    if (day.value.trim() == "") {
        invalidDay();
    }

    if (validDate) {
        const birthDay = new Date(year.value, month.value - 1, day.value);

        let resYear = date.getFullYear() - birthDay.getFullYear();
        let resMonth = 0;
        date.getMonth() > birthDay.getMonth() ? (resMonth = date.getMonth() - birthDay.getMonth() - 1) : (resYear--, resMonth = 12 - (birthDay.getMonth() - date.getMonth()));
        const resDay = 31 - Math.abs(date.getDate() - day.value) - 1;
        if (DEBUG) {
            console.log("DATE.GETMONTH()", date.getMonth());
            console.log("BIRTHDAY.GETMONTH()", birthDay.getMonth());
            console.log("SUBTRACTING", date.getMonth() - birthDay.getMonth());
            console.log("MONTH", resMonth);
            console.log("date.now", Date.now());
            console.log("birthDay", birthDay.getTime());
            console.log("RESULT", resYear, resMonth, resDay);
            console.log(typeof year.value);
            console.log("year, month, day", year.value, month.value, day.value);
        }
        
        document.querySelector('#year').innerHTML = `${resYear}`;
        document.querySelector('#month').innerHTML = `${resMonth}`;
        document.querySelector('#day').innerHTML = `${resDay}`;
        
    }

});