import editIcon from './assets/edit-icon.png'
import questionIcon from './assets/question-icon.png'

function GridParentTitle() {

    return(
        <div className="container">
            <div className="empty"></div>
            <div className="details">
                <div className="gridContainerChild1">
                    <form>
                        <label className="type">Ex Series</label>
                        <input className="name" type="text" defaultValue="Untitled Machine" autoFocus></input>
                    </form>
                    <img className="icon" src={editIcon} alt="edit-icon"></img>
                    <img className="question-icon" src={questionIcon}></img>
                </div> {/* gridContainerChild1 */}
            </div> {/* details */}
            <div className="rightColumn">
                <div className="gridContainerChild2">
                    <p>
                        Resolve invalid values to continue.
                    </p>
                    <button className="reviewButton">
                        Review
                    </button>
                    <button className="reviewMemoryMapButton">
                        Review Memory Map
                    </button>
                </div> {/* gridContainerChild2 */}
            </div> {/* rightColumn */}
        </div>
    );



}

export default GridParentTitle