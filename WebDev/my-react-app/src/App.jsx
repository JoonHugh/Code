import Header from './Header.jsx';
import Footer from './Footer.jsx';
import Progress from './Progress.jsx';
import GridParentTitle from './GridParentTitle.jsx';
import Container from './Container.jsx';
import MainGrid from './MainGrid.jsx';

function App() {
  return(
      <Container>
        <Header />
        <Progress />
        <GridParentTitle />
        <MainGrid />
        <Footer />
      </Container>
  );
}

export default App