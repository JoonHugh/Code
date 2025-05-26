import AC from "./assets/AC.png";
import PosNeg from "./assets/PosNeg.png";
import "./grid.css";

function Grid() {
    return(
        <div className="grid">
            <span><img className="AC" src={AC}></img></span>
            <span><img className="PosNeg" src={PosNeg}></img></span>
        </div>
    );
}

export default Grid