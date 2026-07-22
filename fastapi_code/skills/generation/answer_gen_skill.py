class AnswerGenerateSkill(BaseSkill):


 async def run_skill():

       refs = build_reference()

       llm_response = call_llm()

       return response