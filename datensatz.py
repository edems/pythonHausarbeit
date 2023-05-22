from sqlalchemy import create_engine, Column, Integer, String, Float ,inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


username = 'adam'
password = ''
host = 'hausarbeit.mysql.database.azure.com'
database = 'hausarbeit'
engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{host}/{database}", echo=True)

# Basisklasse für die Tabellendeklaration
Base = declarative_base()

# Tabelle deklarieren
class Moja(Base):
    __tablename__ = 'meinetesttable'
    id = Column(Integer, primary_key=True, autoincrement=True)
    column1 = Column(String(255))
    column2 = Column(Integer)
    column3 = Column(Float)

# Tabelle erstellen, wenn sie noch nicht existiert
Base.metadata.create_all(bind=engine, checkfirst=True)

# Inspektor erstellen
inspector = inspect(engine)


# Verbindung aufbauen und Session verwenden
with engine.connect() as connection:
    # Session erstellen
    Session = sessionmaker(bind=connection)
    session = Session()

    # Weitere Operationen auf der Datenbank durchführen
    # Tabellen in der Datenbank abrufen
    tables = inspector.get_table_names()
    for table in tables:
        print("Meine neue Table")
        print(table)

    result = session.query(Moja).all()

    # Ergebnis ausgeben
    for row in result:
        print(row.id, row.column1, row.column2, row.column3)

    # Verbindung schließen (automatisch durch das 'with'-Statement)
    session.close()