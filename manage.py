from app import app
import os, click
from database.db import db
from flask_script import Manager
from flask_migrate import Migrate
from flask.cli import FlaskGroup
from src.Commands.Migration.classMigrate import Migration
from src.Commands.populateData import addDefaultAdmin, addFeedbackQuestions, addQuestion
from src.Commands.migrateCommand import generate_model_class
from src.Commands.migrateCommandMPstruct import generate_model_class_MPSTRUCT
from src.Commands.migrateCommandPDB import generate_model_class_PDB
from src.Commands.migrateCommandOPM import generate_model_class_OPM
from src.Commands.migrateCommandUniprot import generate_model_class_Uniprot
from src.utils.helpers import clear_file_content

migrate = Migrate(app, db)
manager = Manager(app)

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting file '{file_path}': {e}")

@app.cli.command("sync-protein-database")
@click.option('--clear_db', default="n", help='Do you really want to clear the DB')
def init_migrate_mpstruct_upgrade(clear_db):
    print(clear_db)
    # Path to the directory where migrations will be stored
    migrations_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'migrations')

    if not os.path.exists(migrations_dir):
        with app.app_context():
            click.echo("Running 'Initialize Database Migrations'")
            os.system('flask db init')
            
    if clear_db == "y": # dropp all tables and build again    
        print("Dropping all tables")  
        db.drop_all()
    
    clear_file_content("src/MP/model_mpstruct.py")
    model_class_mpstruct = generate_model_class_MPSTRUCT('./datasets/valid/Mpstruct_dataset.csv', 'src/MP/model_mpstruct.py')
    clear_file_content("src/MP/model_pdb.py")
    model_class_pdb = generate_model_class_PDB('./datasets/valid/PDB_data_transformed.csv', 'src/MP/model_pdb.py')
    clear_file_content("src/MP/model_opm.py")
    model_class_opm = generate_model_class_OPM('./datasets/valid/NEWOPM.csv', 'src/MP/model_opm.py')
    clear_file_content("src/MP/model_uniprot.py")
    model_class_uniprot = generate_model_class_Uniprot('./datasets/Uniprot_functions.csv', 'src/MP/model_uniprot.py')
    clear_file_content("src/MP/model.py")
    model_class = generate_model_class('./datasets/valid/Quantitative_data.csv', 'src/MP/model.py')

    """Initialize, migrate, and upgrade the database."""
    with app.app_context():
        """Initialize, migrate, and upgrade the database."""
        click.echo("Running 'flask db init'")
        click.echo("This step is optional if you have already initialized migrations manually.")
        click.echo("If you haven't, you can run 'flask db init' separately.")
        click.echo("Running 'flask db migrate'")
        os.system('flask db migrate')
        click.echo("Running 'flask db upgrade'")
        os.system('flask db upgrade')
        click.echo("Database initialization, migration, and upgrade completed.")

    db.create_all()   
    # Call the function to load data into the database
    
    Migration.load_csv_data(model_class_mpstruct, './datasets/valid/Mpstruct_dataset.csv')
    Migration.load_csv_data(model_class_pdb, './datasets/valid/PDB_data_transformed.csv')
    Migration.load_csv_data(model_class_opm, './datasets/valid/NEWOPM.csv')
    Migration.load_csv_data(model_class_uniprot, './datasets/Uniprot_functions.csv')
    Migration.load_csv_data(model_class, './datasets/valid/Quantitative_data.csv')
    # add questions and options
    addQuestion()
    # add admin
    addDefaultAdmin()
    # add feedback question
    addFeedbackQuestions()

@app.cli.command("sync-question-with-database")
def init_data_questions():
    # add questions and options
    addQuestion()
    
@app.cli.command("sync-system_admin-with-database")
def init_system_admin():
    addDefaultAdmin()
      
@app.cli.command("sync-feedback-questions-with-database")
def init_system_admin():
    addFeedbackQuestions()
    
if __name__ == '__main__':
    manager.run()
