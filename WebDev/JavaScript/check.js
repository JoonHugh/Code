const button = document.querySelectorAll('button');
const buttonGroup = document.querySelector('.buttonGroup')

button[0].addEventListener('click', click) 
button[1].addEventListener('click', click) 
button[2].addEventListener('click', click) 

const pressed = false;

buttonGroup.addEventListener('click', (event) => {
    if (event.target.classList.contains('button')) {
        
        const clickedButtonValue = event.target.value;
        console.log('clicked button value:', clickedButtonValue);
    }
})



function click(e) {
    e.preventDefault();

    console.log("CLICKED");
}
// console.log(button);
