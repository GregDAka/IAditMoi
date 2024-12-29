import csv

def txt_to_csv(txt_file, csv_file):
    # Ouvrir le fichier texte en mode lecture
    with open(txt_file, 'r') as txt:
        # Lire toutes les lignes du fichier texte
        lines = txt.readlines()

        # Ouvrir le fichier CSV en mode écriture
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Parcourir chaque ligne du fichier texte
            for line in lines:
                # Nettoyer la ligne et la séparer par les virgules
                row = line.strip().split(',')
                
                # Écrire la ligne dans le fichier CSV
                writer.writerow(row)


txt_to_csv('D:/HOME/Desktop/Convertisseur TXT CSV/vehicule.txt', 'vyyhicule.csv')
