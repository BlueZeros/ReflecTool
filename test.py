from reflectool.agents.TaskAgent import TaskAgent
from reflectool.commons.TaskPackage import TaskPackage
from reflectool.utilities import parse_args


args = parse_args()
args.model = "gpt-4o-mini"
Agent = TaskAgent(args)

task = TaskPackage(
    inputs="hello, can you explain me about the fever?",
    instruction="",
)
print(Agent(task))
