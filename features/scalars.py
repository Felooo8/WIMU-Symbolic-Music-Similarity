import muspy

class MusicScalars:
    def __init__(self, measure_resolution: int = 1):
        self.measure_resolution = measure_resolution

        self.pitch_class_entropy = 0.0
        self.pitch_entropy = 0.0
        self.pitch_range = 0
        self.scale_consistency = 0.0
        self.polyphony = 0.0
        self.empty_beat_rate = 0.0
        self.groove_consistency = 0.0
       
    def calc(self, file_path: str):
        music = muspy.load(file_path)

        self.pitch_class_entropy = muspy.pitch_class_entropy(music) 
        self.pitch_entropy = muspy.pitch_entropy(music) 
        self.pitch_range = muspy.pitch_range(music) 
        self.scale_consistency = muspy.scale_consistency(music)
        self.polyphony = muspy.polyphony(music) 
        self.empty_beat_rate = muspy.empty_beat_rate(music)
        self.groove_consistency = muspy.groove_consistency(music, self.measure_resolution) 

        return self.pitch_class_entropy, self.pitch_entropy, self.pitch_range, self.scale_consistency, self.polyphony, self.empty_beat_rate, self.groove_consistency

    def get_as_txt(self):
        return f"metrics: \
                \n\tPitch class entropy: {self.pitch_class_entropy} \
                \n\tPitch entropy: {self.pitch_entropy} \
                \n\tPitch range: {self.pitch_range} \
                \n\tScale consistency: {self.scale_consistency} \
                \n\tPolyphony: {self.polyphony} \
                \n\tEmpty beat rate: {self.empty_beat_rate} \
                \n\tGroove consistency: {self.groove_consistency} with measure_resolution={self.measure_resolution}"