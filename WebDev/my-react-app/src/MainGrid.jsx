
import React, {useState} from 'react';
import questionIcon from './assets/question-icon.png';

function MainGrid() {

    const tabs = [{id: "tab1", name: "Modes & ISA"}, {id: "tab2", name: "On-Chip Memory"}, {id: "tab3", name: "MMIO Base Addr"}, 
                  {id: "tab4", name: "Ports"}, {id: "tab5", name: "Security"}, {id: "tab6", name: "Debug & Trace"}, {id: "tab7", name: "Interrupts"}, 
                  {id: "tab8", name: "Design For Test"}, {id: "tab9", name: "Clocks and Reset"}, {id: "tab10", name: "Power"}, 
                  {id: "tab11", name: "Branch Prediction"}, {id: "tab12", name: "World Guard"}, {id: "tab13", name: "Other Default IPs"}, 
                  {id: "tab14", name: "RTL Options"}
                 ];

    const tabItems = tabs.map(tab => <li key={tab.name} className="generic-styled-tab">{tab.name}</li>)

    const [tab, setTab] = useState();
    // const []

    const handleClick = (e) => {
        const siblings = e.target.parentElement.children;
        e.target.parentElement.style.borderLeft = "0px solid transparent";
        for (let el of siblings) {
            el.style.borderLeft = "5px solid transparent";
            el.style.color = "hsl(0, 0%, 55%)";
        }
        console.log(e);
        // e.target.parentElement.children.style.borderLeft = "5px solid transparent";
        e.target.style.borderLeft = "5px solid hsl(14, 100%, 55%)";
        e.target.style.color = "hsl(0, 0%, 35%)";
    }


    return(
        <div className="main-grid">
            <div className="left-column">
                <ul onClick={handleClick} className="tab-list">{tabItems}</ul>
            </div>
            <div className="middle-column">
                <h2>Modes & ISA</h2>
                <div className="tab-section">
                    <label>
                        <span>Total Number of Cores</span>
                        <img src={questionIcon} className="tab-section-question-icon"></img>
                        <input className="tab-section-input" type="range" min="1" max="4" step="1" list="steplist" />
                    </label>
                    <datalist id="steplist">
                        <option value="1"/>
                        <option value="2"/>
                        <option value="3"/>
                        <option value="4"/>
                    </datalist>
                    <div className="ticks">
                        <div className="tick"></div>
                        <div className="tick"></div>
                        <div className="tick"></div>
                        {/* One less tick div than number of steps */}
                    </div>
                </div> {/* tab section */}
                <div className="tab-section">
                    <span>Privilege Modes</span>
                    <label className="tab-section-label" id="disabled" disabled>Machine Mode
                        <input className="tab-section-checkbox" type="checkbox" checked disabled></input>
                        <span className="checkmark"></span>
                    </label>
                    <label className="tab-section-label">User Mode
                        <input className="tab-section-checkbox" type="checkbox"></input>
                        <span className="checkmark"></span>
                    </label>
                </div> {/* tab section */}
                <div className="tab-section">
                    <span>Base ISA</span>
                    <div class="buttonGroup">
                        <button class="button" value="RV32I">RV32I</button><button class="button" value="RV32E">RV32E</button>
                    </div>
                </div> {/* tab section */}
                <div className="tab-section">
                    <span>ISA Extensions</span>
                    <label className="tab-section-label"> Code density extensions
                        <input className="tab-section-checkbox" type="checkbox"></input>
                        <span className="checkmark"></span>
                    </label>
                    <label className="tab-section-label"> Multiply (M Extension)
                        <input className="tab-section-checkbox" type="checkbox"></input>
                        <span className="checkmark"></span>
                    </label>
                    <div>

                    </div>
                    <span>Floating Point</span> 
                    <img src={questionIcon} className="tab-section-question-icon"></img>
                    <div class="buttonGroup">
                        <button class="button" value="No-FP">No FP</button><button class="button" value="Single-FP">Single FP (F)</button><button class="button" value="Double-FP">Double FP (F & D)</button>
                    </div>
                </div> {/* tab section */}
            {/* </div> */}
            </div>
            <div className="right-column">
                <div className="box">
                    <h5>Untitled Machine Core Complex</h5>
                    <div>
                        <h6>??? Series Core</h6>
                        <h6>? Cores</h6>
                        <h6>JA920IMAC</h6>
                    </div>
                        <span>____ Mode</span>
                        <span>User Mode</span>

                    <div>

                    </div>
                </div>
            </div>
        </div>
    );
}

export default MainGrid