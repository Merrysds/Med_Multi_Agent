# skills/base.py
from pydantic import BaseModel

class BaseSkillInput(BaseModel):
    """所有Skill入参基类"""
    pass

class BaseSkillOutput(BaseModel):
    """所有Skill出参基类"""
    pass

class BaseSkill:
    """Skill统一父类，强制标准化接口"""
    async def run_skill(self, input: BaseSkillInput) -> BaseSkillOutput:
        raise NotImplementedError("子类必须实现 run_skill 方法")