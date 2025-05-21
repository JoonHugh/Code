import React, { useState } from 'react';

function MainGrid({ items }) {

    const [clickedItem, setClickedItem] = useState(null);
    
    const handleItemClick = (item) => {
        setClickedItem(item);
    };

    return(
        <div className="main-grid">
            <div className="left-column">
                <ul className="tab-list">
                    {}
                        <li className="generic-styled-tab">Modes & ISA</li>
                        <li className="generic-styled-tab">On-Chip Memory</li>
                        <li className="generic-styled-tab">MMIO Base Addr</li>
                        <li className="generic-styled-tab">Ports</li>
                        <li className="generic-styled-tab">Security</li>
                        <li className="generic-styled-tab">Debug & Trace</li>
                        <li className="generic-styled-tab">Interrupts</li>
                        <li className="generic-styled-tab">Design For Test</li>
                        <li className="generic-styled-tab">Clocks and Reset</li>
                        <li className="generic-styled-tab">Power</li>
                        <li className="generic-styled-tab">Branch Prediction</li>
                        <li className="generic-styled-tab">World Guard</li>
                        <li className="generic-styled-tab">Other Default IPs</li>
                        <li className="generic-styled-tab">RTL Options</li>
                </ul>
            </div>
            <div className="middle-column">
                <h2>Modes & ISA</h2>
                <div className="core-count">
                    <label>
                        Total Number of Cores
                        <input type="range" min="1" max="4" step="1" list="steplist" />
                    </label>
                    <datalist id="steplist">
                        <option value="1"/>
                        <option value="2"/>
                        <option value="3"/>
                        <option value="4"/>
                    </datalist>
                </div>
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