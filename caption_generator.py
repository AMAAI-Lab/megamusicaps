from openai import OpenAI

class CaptionGenerator:
    def __init__(self, api_key, model_id):
        self.api_key = api_key
        self.model_id = model_id

        self.client = OpenAI(api_key = self.api_key) 

        # self.prompt_base = "I am creating a dataset for a prompt based music generation system. Generate a prompt that can be used as the label for a music snippet that has the following features. For the chord features, could you summarise the chord progression generally without too much detail and mention any trends or special points that you observe. Ignore all the No Chords (N):\n"

        self.prompt_base = "I am trying to create a dataset that has an audio file path, a query that contains some musical information about the audio file, and a ground truth caption for the audio. This dataset would be used to then train a model that can caption music.\n Here are some examples of the queries and the paired captions. \n\n mood: happy, genre: classical, instrument: [cello, claps, drums, bass], autotags: [instrumental], voice: instrumental, beats: {bpm: 85.10238907849829, beat_pattern: [1.0, 2.0, 3.0, 4.0]}, chords: [0.000 6.759 N, 6.759 10.000 C, 10.000 16.111 D, 16.111 17.037 E, 17.037 18.426 D, 18.426 20.000 E, 20.000 21.019 C, 21.019 24.259 G, 24.259 30.000 N], gender: female}\n question: This audio is in the A key. The tempo is 85 bpm and the beat is a 4/4 beat. It has a happy mood and is an instrumental. It is in the classical music genre. The chords in the song are C, D, E, G. It includes cello, claps, drums and bass. Please create a caption for this music clipping,\n answer : This song features two cellos playing a melody in harmony. This is accompanied by hand claps on every alternate count of the bars. The bass plays the root notes of the chords. The kick drum is played at the intro. There are no voices in this song. This is an instrumental song with a happy feel. This song can be played in a luxury advertisement. \n\n mood: melancholic, genre: oriental, instrument: [harp], autotags: [fast, live], voice: instrumental, beats: {bpm: 93.8394707849829, beat_pattern: [1.0, 2.0, 3.0, 4.0]}, chords: [0.000 6.759 N, 6.759 10.000 Am, 10.000 16.111 C, 16.111 17.037 D, 17.037 18.426 Fmin, 18.426 20.000 Am, 20.000 21.019 C, 21.019 24.259 D, 24.259 30.000 Fmin], gender: female}\n question: This music has a harp. It has a tempo of 93 bpm. The genre would be oriental music and the mood is melancholic. It is fast, live music. The chords used are Am, C, D, Fmin. How would you describe this audio clip?, \n answer: Someone is playing a harp making use of the full register. A bassline, a melody is the mid range and high repetitive note. The whole composition sounds oriental due to a lot of half notes. This song may be playing at a live concert. \n\n mood: energetic, genre: hip-hop, instrument: [piano, drums, synth bass, vocals], autotags: [repetitive], voice: female, beats: {bpm: 120.4958907849829, beat_pattern: [1.0, 2.0, 3.0, 1.0 ,2.0, 3.0, 4.0]}, chords: [0.000 6.759 A, 6.759 10.000 G, 10.000 16.111 A, 16.111 17.037 G, 17.037 18.426 A, 18.426 20.000 G, 20.000 21.019 A, 21.019 24.259 G, 24.259 30.000 N], gender: male} \n, question: This is a hip-hop song. It uses a piano, drums, synth bass and vocals. The gender of the vocals is male. It is energetic. The music is repetitive. The tempo is 120 bpm. The chords are A and G. Create a caption for it,\n answer: The Hip Hop song features a flat male vocal rapping over repetitive, echoing piano melody, claps and punchy kick hits. At the very beginning of the loop, there are stuttering hi hats, sustained synth bass and filtered, pitched up female chant playing. It sounds groovy and addictive - thanks to the way rappers are rapping.\n\n\n Note how the query does not information provided, while the caption may do that. \n\n\n here is the information for an audio clip. Can you please create the question for it? Make sure to include all the information I have provided and don't stray from it. \n"

    def set_prompt_base(self, prompt_base):
        self.prompt_base = prompt_base
        if self.prompt_base[-1] != '\n':
            self.prompt_base += '\n'

    def create_prompt(self, keywords):
        prompt = self.prompt_base
        prompt += str(keywords)

        return prompt

    def generate_caption(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_id,
        )
        caption = response.choices[0].message.content
        return caption
