import styled from 'styled-components';

const Wrapper = styled.div`
    max-width: 95vw;
    height:100vh;
    margin: 0 auto;
    display:flex;
    flex-direction:column;
`

function Container({ children }) {


    return(
        <Wrapper>
            {children}
        </Wrapper>
    );

}

export default Container