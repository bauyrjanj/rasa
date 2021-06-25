# 0. Objective 
The purpose of this submission is to demonstrate the technical skills expected of a candidate for Solutions engineer position. 

# 1. Scope of the exercise
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


# 2. Development

## 2.1 Installation of Rasa Open Source
```
# First create a dedicated directory for the project under the current working directory
mkdir rasa_take_home_exam

# Second, cd into the newly created project directory
cd rasa_take_home_exam

# Third, it is a good practice to create a dedicated virtual environment for the project to avoid interfering with the packages for other projects
# In this case, I create a virtual environment called rasa_bot with python version 3.7.5
conda create --name rasa_bot python==3.7.5

# Next, activate the newly created virtual environment
conda activate rasa_bot

# Install ultra json as we need it for the bot
conda install ujson

# Install tensorflow
conta install tensorflow

```

Before installing rasa, I had to install or update the Visual Studio Buld Tools.

Install Visual Studio Build Tools >=14.0
    * MS Visual Studio C++
	* Windows SDK 

Finally, installed rasa with the following code. 
```
pip install rasa==1.10.12

pip install rasa_sdk==1.10.1
```

## 2.2 Challenges faced during installation

I encountered the following error while attempting to install rasa version 1.10.12

    ERROR: Could not build wheels for ujson which use PEP 517 and cannot be installed directly
	
## 2.3 How I solved the above challenge
 
I had to update the Visual Studio Build Tools on my local machine and it solved the above problem

# 3. Build 

I built the initial prototype of the rasa bot model with the below code. 

```
rasa init

```


# 4. Training the bot 

I started by creating some training data for nlu (intent examples), nlu core (stories) and also created entities, slots, responses in the domain.yml file. I also listed the custom actions in the actions section of the domain.yml. I created 6 intents and 11 stories in total to icnlude in the training set.

I am using regex to extract names from the conversations and also used 5 slot variables to store information as required. There are 2 entities that are extracted from the conversations and are also stored in slots to be either consumed by the custom actions or stories. 

Then I worked on the pipeline in the config.yml file. I tried to implement and compare two different pipelines, both leveraging a pre-trained language model as I didn't have enough training data.  Find more details below.

I also uncommented the action_endpoint in the endpoints.yml file as I need it to run my custom action server. 

In general, I used the following commands the most often during the training phase of the model/bot.  

  ```
  # train the mode
  rasa train
  
  # run the model in the shell
  rasa shell
  
  # run custom actions
  rasa run actions

  ```

## 4.1 Challenges faced

I face the following error when trying to load the model with "rasa shell" 

    ERROR:  Could not load model due to Domain specification has changed. You MUST retrain the policy. Detected mismatch in domain specification. The following states have been
            - removed: slot_status_4
            - added:    .

## 4.2 How I solved the above challenge:

Made changes to the order of components in the pipeline and also changed the types (text vs categorical etc.) of the slots. After making these changes, the error disappeared. I am not 100% sure what the cause of this error but 
I suspect it has something to do with the categorical slots where values are listed and the max_history of the policies. I never encountered this issue with the rasa==2.5.0 before so this was new to me.

## 4.3 Decisions 

### 4.3.1 Pipeline
  
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

### 4.3.2 Custom actions
   
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
   * Handle the synonymous slot values for position_type (i.e. tech vs technical, any vs anything etc.)
   * Correctly list the jobs by using "," and "and" appropriately (i.e. "ML engineer and Solutions Engineer" vs "ML engineer, Software developer, and Solutions Engineer")
   * Return an additional slot value, role_type, back to the core to use in the conversation path 
   
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

### 4.3.3 NLU
   
  * Used the slot "role_types" in the stories regarding the inquiry of the open positions to provide the correct response about the types of open positions based on the value of this slot variable.
   
  * Used the slot "status" in the stories regarding the job application status to provide the correct response about the actual status of the job application (received vs rejected vs interview etc.) based on the value of this slot variable.
   
  * Used the slot "positions" in the responses (in domain.yml) to respond back to the users with the list of open positions.
   
  * Used the slot "name" in the responses (in domain.yml) to respond back to the users by addressing them with their names.
   
# 5. Testing

```
rasa test nlu -u data/nlu.yml --config config.yml --cross-validation

```

Please find the results of the test in the results directory of this repo. For the summary report of the intents, please see the file "intent_report.md" or run "python report_results.py" while in the results directory.


# 6. Rasa X

Although it is not required for this exercise, I have deployed Rasa X and uploaded the model for interactive learning. It is a perfect way to improve the assistance. Please click on the below link to interact with the model in an interactive learning mode.

http://35.197.29.212:8000/guest/conversations/production/c067895460fb4c7f84cfd1728f2d24f4

