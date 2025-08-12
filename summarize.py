import pickle
from semantic_text_splitter import TextSplitter
from openai import OpenAI



"""completion = client.chat.completions.create(
  #model = "qwen/qwen3-8b",
  model="gemma-1.1-2b-it",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ],
  temperature=0.7,
)"""

def test_connection():
    client = OpenAI(base_url="http://192.168.178.25:1234/v1", api_key="lm-studio")
    completion = client.chat.completions.create(
      #model = "qwen/qwen3-8b",
      model="gemma-1.1-2b-it",
      messages=[
        {"role": "system", "content": "Always answer in json."},
        {"role": "user", "content": "Generate some json data that describes you as a system."}
      ],
      temperature=0.7,
    )

    print(completion.choices[0].message)

def summarize():
    with open('data/out/pkl/final_chunks.pkl', 'rb') as chunks:
        final_chunks = pickle.load(chunks)
    with open('data/out/pkl/numbering.pkl', 'rb') as numbering:
        numbering = pickle.load(numbering)

    chunk_summary_chunk_number = []
    chunk_number = []
    chunk_summaries = {}
    chunk_length_before_after=[]

    max_characters = 512
    splitter = TextSplitter(max_characters, trim=False)
    client = OpenAI(base_url="http://192.168.178.25:1234/v1", api_key="lm-studio")
    print("connecting...")
    print(type(final_chunks))
    for i,(chunk_number, chunk)  in enumerate(final_chunks.items()):
        chunk_length_before=len(chunk)

        completion = client.chat.completions.create(
            # model = "qwen/qwen3-8b",
            model="gemma-1.1-2b-it",
            messages=[
                {
                    "role": "user",
                    "content": "Summarize the following text in a technical way. Focus on facts, numbers and strategies used. Add a title to the summary. Be impersonal:\n\n{}".format(chunk)
                }
            ],
            temperature=0.4,
        )
        print("answer")
        answer = completion.choices[0].message.content
        chunk_summaries[chunk_number]=answer
        chunk_length_after = len(answer)
        chunk_length_before_after.append((chunk_length_before,chunk_length_after))

    client.close()
    print("finished")
    with open('data/out/pkl/chunk_summaries.pkl', 'wb') as f:
        pickle.dump(chunk_summaries, f)

    with open('data/out/pkl/chunk_length_ba.pkl', 'wb') as f:
        pickle.dump(chunk_length_before_after, f)


        """chunks_summaries = splitter.chunks(answer)
        count=1
        for c in chunks_summaries:
            print(c)
            print(len(c))
            chunk_number.append(chunk_number)
            chunk_summary_chunk_number.append(str(count))
            
            count=count+1"""

    #re=["/n"]
    #result = answer.replace("/n", "")



    #print(result)
    #print(len(result))
    final_summaries=[]

def get_from_pkl():
    with open('data/out/pkl/final_chunks.pkl', 'rb') as chunks:
        final_chunks = pickle.load(chunks)
    with open('data/out/pkl/chunk_summaries.pkl', 'rb') as chunk_summa:
        chunk_summaries = pickle.load(chunk_summa)
    with open('data/out/pkl/numbering.pkl', 'rb') as numbering:
        numbering = pickle.load(numbering)
    with open('data/out/pkl/chunk_length_ba.pkl', 'rb') as chunk_length_before_after:
        chunks_length=pickle.load(chunk_length_before_after)

    x, y = zip(*chunks_length)

    res1 = (min(x), max(x))  # first
    res2 = (min(y), max(y))  # second

    print("Lowest and highest length chunks:")
    print(res1)
    print("Lowest and highest length summaries:")
    print(res2)



    #print(final_chunks)
    """for (k,v) in enumerate(final_chunks):
        #print(k)
        val=final_chunks.get(k)
        #print(v)
        print("length")
        print(val)
        return"""
    ch = {'‡', '§', '†', '\n'}
    for i,(k,v) in enumerate(final_chunks.items()):
        print(v)
        print("##############")
        s2 = "".join([c for c in v if c not in ch])
        print(s2)
        break

    for i,(k,v) in enumerate(chunk_summaries.items()):
        print(len(v))
        print(v)
        break





