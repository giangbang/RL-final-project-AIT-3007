import collections
import random

class TeamManager:

    def __init__(self, agents, my_team = None):
        self.agents = agents
        self.teams = self.group_agents()
        self.terminated_agents = set()
        self.my_team = my_team
        self.random_agents = None
        self.get_random_agents(1)

    def get_teams(self):
        """
        Get the team names.
        :return: a list of team names
        """
        return list(self.teams.keys())

    def get_my_team(self):
        if self.my_team is not None:
            return self.my_team
        else:
            my_team = self.get_teams()[1]
        self.my_team = my_team
        return my_team

    def get_other_team(self):
        return self.get_teams()[0]

    def get_team_agents(self, team):
        """
        Get the agents in a team.
        :param team: the team name
        :return: a list of agent names in the team
        """
        assert team in self.teams, f"Team [{team}] not found."
        return self.teams[team]

    def get_my_agents(self):
        return self.get_team_agents(self.get_my_team())

    def group_agents(self):
        """
        Group agents by their team.
        :param agents: a list of agent names in the format of teamname_agentid
        :return: a dictionary with team names as keys and a list of agent names as values
        """
        teams = collections.defaultdict(list)
        for agent in self.agents:
            team, _ = agent.split('_')
            teams[team].append(agent)
        return teams

    def get_info_of_team(self, team, data, default=None):
        """
        Get the information of a team.
        :param team: the team name
        :param data: the data to get information from
        :return: a dictionary with the team name as key and the information as value
        """
        assert team in self.teams, f"Team [{team}] not found."
        result = {}
        for agent in self.get_team_agents(team):
            if agent not in data:
                result[agent] = default
            else:
                result[agent] = data[agent]
        return result
    
    def reset(self):
        self.terminated_agents = set()

    def is_team_terminated(self, team):
        """
        Check if all agents in a team are terminated.
        :param team: the team name
        :return: True if all agents in the team are terminated, False otherwise
        """
        assert team in self.teams, f"Team [{team}] not found."
        return all(agent in self.terminated_agents for agent in self.teams[team])

    def terminate_agent(self, agent):
        """
        Mark an agent as terminated.
        :param agent:
        :return:
        """
        self.terminated_agents.add(agent)

    def has_terminated_teams(self):
        """
        Check if any team is terminated.
        """
        for team in self.teams:
            if self.is_team_terminated(team):
                return True
        return False

    def get_my_terminated_agents(self):
        return list(self.terminated_agents.intersection(self.get_my_agents()))

    def get_other_team_remains(self):
        """
        Get the remaining agents in the other team.
        :return:
        """
        my_team = self.get_my_team()
        other_team = [team for team in self.teams if team != my_team][0]
        return [agent for agent in self.get_team_agents(other_team) if agent not in self.terminated_agents]


    def get_random_agents(self, rate):
        """
        Create a random agent list, and return the first n agents.
        :param rate: the rate of random agents to return
        :return: a list of random agents with the length of rate * num_agents
        """
        num_agents = len(self.get_my_agents())
        if self.random_agents is not None:
            num_random_agents = int(num_agents * rate)
            return self.random_agents[:num_random_agents]
        else:
            self.random_agents = random.sample(self.get_my_agents(), num_agents)
            return self.get_random_agents(rate)

    @staticmethod
    def merge_terminates_truncates(terminates, truncates):
        """
        Merge terminates and truncates into one dictionary.
        :param terminates: a dictionary with agent names as keys and boolean values as values
        :param truncates: a dictionary with agent names as keys and boolean values as values
        :return: a dictionary with agent names as keys and boolean values as values
        """
        result = {}
        for agent in terminates:
            result[agent] = terminates[agent] or truncates[agent]
        return result