class Match(db.Model):
    id = db.Column(db.String(50), primary_key=True)  # ID pertandingan
    team_a = db.Column(db.String(50), nullable=False)  # Nama tim A
    team_b = db.Column(db.String(50), nullable=False)  # Nama tim B
    result = db.Column(db.String(50))  # Hasil akhir pertandingan

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    match_id = db.Column(db.String(50), db.ForeignKey('match.id'), nullable=False)  # Referensi ke pertandingan
    chosen_team = db.Column(db.String(50), nullable=False)
    is_correct = db.Column(db.Boolean, default=False)
