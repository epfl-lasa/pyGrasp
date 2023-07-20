from dataclasses import dataclass

from pyGrasp.robot_model import RobotModel

"""
 Here is what we do for a reachable spaces:

    
    # Resolve all links
    current_link = base_link
    while base_link not resolved:
        
        if current_link not resolved 
            if not current_link.all_descendence_resolved:
                current_link = furthest_unresolved_descendent
            else:
                compute current link reachable space as alpha shape
                propagate link movement on the children 
    
    
    # Resolve and propagate link
    for angle in joint_angle_list:
        move_robot to position with every angle to 0 but angle
        get mesh of current node
        get mesh of children at their place
        
                        
"""


class ReachableSpaces:
    
    @dataclass
    class LinkInfo:
        name: str
        id: int
        rs_solved: bool
        children_id: list[int]
        children_names: list[str]
        
    def __init__(self, robot_model: RobotModel) -> None:
        self.robot_model = robot_model
        self._link_map = {}   # Key = link, value = children
        
    def compute(self) -> None:
        
        # Generate link map
        if not self._link_map:
            self._create_link_map()
        
    def _create_link_map(self) -> None:
        """Generate a dictionnary of links and their children to be able to brows them from base to tip
        """
        for i, link in enumerate(self.robot_model.links):
            
            # Add link to the dictionary if needed
            if link.name not in self._link_map:
                self._link_map[link.name] = ReachableSpaces.LinkInfo(name=link.name,
                                                                     id=i,
                                                                     rs_solved=False,
                                                                     children_id=[],
                                                                     children_names=[])
            # Means we added it as a parent and couldn't fill the id yet so we fix it now
            else:
                self._link_map[link.name].id = i
                
            # Add parent to the dictionary if needed
            if link.parent is not None:
                if link.parent.name not in self._link_map:
                    self._link_map[link.parent.name] = ReachableSpaces.LinkInfo(name=link.parent.name,
                                                                                id=-1,
                                                                                rs_solved=False,
                                                                                children_id=[i],
                                                                                children_names=[link.name])
        
                # Append to children if parent already in dict
                else:
                    self._link_map[link.parent.name].children_id.append(i)
                    self._link_map[link.parent.name].children_names.append(link.name)
