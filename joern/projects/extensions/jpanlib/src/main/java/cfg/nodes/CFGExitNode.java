package cfg.nodes;

import databaseNodes.NodeKeys;
import java.util.Map;

public class CFGExitNode extends AbstractCFGNode
{

	@Override
	public String toString()
	{
		return -1 != getNodeId() ? "[(" + getNodeId() + ") EXIT]" : "[EXIT]";
	}

	@Override
	public Map<String, Object> getProperties()
	{
		Map<String, Object> properties = super.getProperties();
		properties.put(NodeKeys.CODE, "EXIT");
		return properties;
	}

}
