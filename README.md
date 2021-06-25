### Objective 
The purpose of this submission is to demonstrate the technical skills expected of a candidate for Solutions engineer position. 

### Scope of the exercise
The scope of the exercise is to build a bot that handles the following two kinds of conversations.

The bot should allow for two kinds of conversations:

> “Hi”

“hi, I’m Rasa’s recruiting bot. How can I help?“

> “I’d like to know which positions are open right now”

“Are you looking for a technical or a business role?”

> "A technical one"

“ML Engineer and Solutions Engineer are the open positions.”

**and** 

> “Hi, my name is Ali Park. I applied for a job and would like to know when I’ll hear back”

“Hi Ali! Let me check that for you”

“Yes, your application has been received.”


### Development

## Installation of Rasa Open Source
```
mkdir rasa_take_home_exam

cd rasa_take_home_exam

conda create --name rasa_bot python==3.7.5

conda activate rasa_bot

conda install ujson

conta install tensorflow

```

Install Visual Studio Build Tools >=14.0
    - MS Visual Studio C++
	- Windows SDK 

```
pip install rasa==1.10.12

pip install rasa_sdk==1.10.1
```

# Challenges faced
    ERROR: Could not build wheels for ujson which use PEP 517 and cannot be installed directly
# Solution: 
    Correctly installing the Visual Studio Build Tools solved the above problem

## Build initial prototype of the rasa bot

I build the initial prototype of the rasa bot model with the below code. 

```
rasa init

```


## Training

I used the following commands the most often during the training phase of the model 

  ```
  # train the mode
  rasa train
  
  # run the model in the shell
  rasa shell
  
  # run custom actions
  rasa run actions

  ```

# Challenges faced
    ERROR:  Could not load model due to Domain specification has changed. You MUST retrain the policy. Detected mismatch in domain specification. The following states have been
            - removed: slot_status_4
            - added:    .
# Solution:
    Made changes to the order of components in the pipeline and also changed the types of the slots. After making these changes, the error disappeared. I never encountered this issue with the rasa==2.5.0 so this was new.

# Decisions 
  ** Pipeline
  
  At first, I tried the following pipeline. The reason for that was it is a recommended pipeline for a simple bot that is in english language and have very little data. But the I couldn't get it to work because I needed tensorflow_text 
  and I couldn't get the tensorflow_text work with the rasa 1.10.12 and tensorflow 2.1.0. 
  
  ```
  pipeline: 
  - name: ConveRTTokenizer # breaks down the sentences in tokens aka tokenization
  - name: ConveRTFeaturizer # provides pre-trained word embeddings in English language
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
  
  ```

So I decided to switch to the following pipeline which is another pipeline that is also suitable for a simple bot that is in english language and have very little data. This one works as expected!

   ```
  pipeline:
  - name: SpacyNLP
    model: "en_core_web_lg"
  - name: SpacyTokenizer # breaks down the sentences in tokens aka tokenization
  - name: SpacyFeaturizer # provides pre-trained word embeddings in English language
  - name: RegexFeaturizer
  - name: CRFEntityExtractor
  - name: DIETClassifier
    epochs: 100
    entity_recognition: False
    constrain_similarities: true # this should help to better generalization to real world test sets
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
    constrain_similarities: true
   ```

   ** Custom actions
   
   Made the the appropriate change to the following custom action to handle a situation where user doesn't enter his/her name. The response to the user doesn't address the person with their names.
   
   ```
   class ActionCheckStatus(Action):

    def name(self):
        return "action_check_status"

    def run(self, dispatcher, tracker, domain):
        # return a random status, just a mockup
        statuses = ["received", "rejected", "interview", "unknown"]
        status = random.choice(statuses)

        # get the name of the user from the slot
        name = tracker.get_slot("name")
        if not name:
            status = "noname"
        return [SlotSet("status", status)]
   ```
   
   Made the appropriate changes in "action_check_positions" to handle the following situations:
    - Handle the synonymous slot values for position_type (i.e. tech vs technical, any vs anything etc.)
	- Correctly list the jobs by using "," and "and" appropriately (i.e. "ML engineer and Solutions Engineer" vs "ML engineer, Software developer, and Solutions Engineer")
	- Return an additional slot value, role_type, back to the core to use in the conversation path 
   
   ```
   class ActionCheckPositions(Action):

    def name(self):
        return "action_check_positions"

    def run(self, dispatcher, tracker, domain):
        # return hard-coded open positions, this would normally come from an API
        positions = {
            "technical": [
                "Machine learning engineer",
                "ML product success engineer"
            ],
            "business": []
        }
        position_type = tracker.get_slot("role_type")
        technical_roles = ["tech", "technical", "backend", "engineering"]
        any_roles = ["any", "anything"]

        # Handle the synonymous slot values for position_type
        if position_type in any_roles:
            position_type = "any"
            relevant_positions = positions["technical"] + positions["business"]
        elif position_type in technical_roles:
            position_type = "technical"
            relevant_positions = positions.get(position_type, [])
        else:
            position_type = "business"
            relevant_positions = positions.get(position_type, [])

        # Handle the appropriate listing of the jobs in the response
        if len(relevant_positions)<2:
            relevant_positions = ''.join(relevant_positions)
        elif len(relevant_positions) == 2:
            relevant_positions = ' and '.join(relevant_positions)
        else:
            relevant_positions = ', '.join(relevant_positions[:-1]) + " and " + relevant_positions[-1]
        return [SlotSet("positions", relevant_positions), SlotSet("role_types", position_type)]
   
   ```

   ** NLU
   
   Used the slot "role_types" in the stories regarding the inquiry of the open positions to provide the correct response about the types of open positions based on the value of this slot variable.
   
   Used the slot "status" in the stories regarding the job application status to provide the correct response about the actual status of the job application (received vs rejected vs interview etc.) based on the value of this slot variable.
   
   Used the slot "positions" in the responses (in domain.yml) to respond back to the users with the list of open positions.
   
   Used the slot "name" in the responses (in domain.yml) to respond back to the users by addressing them with their names.
   
## Testing

```
rasa test nlu -u data/nlu.yml --config config.yml --cross-validation

```

Please find the results of the test in the results directory of this repo. 

